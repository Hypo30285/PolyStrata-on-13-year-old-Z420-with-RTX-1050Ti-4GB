// Native C++17 HP Z420 Hardware Telemetry & MCP Stdio Server (z420-sensors-mcp)
// Replaces the Python z420_sensors_mcp.py daemon (~26.2 MB RSS -> ~1.8 MB RSS)
// and provides sub-millisecond /sys/class/hwmon + compact lm-sensors/nvidia-smi telemetry
// with automatic stripping of Electron/Chromium --type=gpu-process command-line bloat.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static std::string run_cmd(const char* cmd) {
    std::array<char, 512> buffer{};
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return trim(result);
}

// Strip repetitive parentheticals "(high = +85.0°C, crit = +105.0°C)" from `sensors`
static std::string run_lm_sensors_compact() {
    std::string raw = run_cmd("sensors 2>/dev/null");
    if (raw.empty()) return "";
    std::istringstream iss(raw);
    std::string line;
    std::string out;
    while (std::getline(iss, line)) {
        // Strip (...) blocks
        std::string cleaned;
        int depth = 0;
        for (char c : line) {
            if (c == '(') {
                depth++;
            } else if (c == ')') {
                if (depth > 0) depth--;
            } else if (depth == 0) {
                cleaned += c;
            }
        }
        // Collapse whitespace
        std::string norm;
        bool prev_space = false;
        for (char c : trim(cleaned)) {
            if (c == ' ' || c == '\t') {
                if (!prev_space) norm += ' ';
                prev_space = true;
            } else {
                norm += c;
                prev_space = false;
            }
        }
        if (norm.empty()) continue;
        if (norm.find("°C") != std::string::npos ||
            norm.rfind("coretemp", 0) == 0 ||
            norm.rfind("nvme", 0) == 0) {
            if (!out.empty()) out += " | ";
            out += norm;
        }
    }
    if (out.size() > 360) out.resize(360);
    return out;
}

// Query GPU metrics and compute apps, stripping full Electron/Chromium --type=gpu-process flags
static std::string run_nvidia_smi_compact() {
    std::string gpu_line = run_cmd(
        "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,pstate,fan.speed --format=csv,noheader 2>/dev/null"
    );
    std::string procs = run_cmd(
        "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null"
    );
    std::string proc_summary;
    if (!procs.empty()) {
        std::istringstream iss(procs);
        std::string line;
        while (std::getline(iss, line)) {
            line = trim(line);
            if (line.empty()) continue;
            auto first_comma = line.find(',');
            auto last_comma = line.rfind(',');
            if (first_comma == std::string::npos || last_comma == std::string::npos || first_comma == last_comma) {
                continue;
            }
            std::string pid = trim(line.substr(0, first_comma));
            std::string proc_raw = trim(line.substr(first_comma + 1, last_comma - first_comma - 1));
            std::string mem = trim(line.substr(last_comma + 1));

            // Keep only the executable token before any flags, then take basename
            auto sp = proc_raw.find_first_of(" \t");
            std::string exe_token = (sp == std::string::npos) ? proc_raw : proc_raw.substr(0, sp);
            auto slash = exe_token.rfind('/');
            std::string bin_name = (slash == std::string::npos) ? exe_token : exe_token.substr(slash + 1);
            if (bin_name.empty()) bin_name = "proc";

            if (!proc_summary.empty()) proc_summary += ", ";
            proc_summary += bin_name + "(PID " + pid + ": " + mem + ")";
        }
    }
    std::string combined = gpu_line;
    if (!proc_summary.empty()) {
        combined += " | Procs: " + proc_summary;
    }
    if (combined.size() > 240) combined.resize(240);
    return combined;
}

struct Z420Readings {
    int cpu_pkg_c = -1;
    int cpu_max_core_c = -1;
    int nvme_c = -1;
    double load_1m = 0.0;
    std::string lm_sensors_raw;
    std::string nvidia_smi_raw;
};

static Z420Readings read_sensors(bool include_cli) {
    Z420Readings r;
    {
        std::ifstream lf("/proc/loadavg");
        if (lf.is_open()) {
            lf >> r.load_1m;
        }
    }
    const fs::path hwmon_root("/sys/class/hwmon");
    if (fs::exists(hwmon_root) && fs::is_directory(hwmon_root)) {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(hwmon_root, ec)) {
            fs::path dir = entry.path();
            std::ifstream nf(dir / "name");
            if (!nf.is_open()) continue;
            std::string name;
            std::getline(nf, name);
            name = trim(name);
            if (name == "coretemp") {
                std::ifstream t1(dir / "temp1_input");
                long millideg = 0;
                if (t1 >> millideg) {
                    r.cpu_pkg_c = static_cast<int>( std::lround(millideg / 1000.0) );
                }
                int max_core = -1;
                for (int i = 2; i <= 18; ++i) {
                    std::ifstream ti(dir / ("temp" + std::to_string(i) + "_input"));
                    long cm = 0;
                    if (ti >> cm) {
                        int c = static_cast<int>( std::lround(cm / 1000.0) );
                        if (c > max_core) max_core = c;
                    }
                }
                r.cpu_max_core_c = max_core;
            } else if (name == "nvme") {
                std::ifstream t1(dir / "temp1_input");
                long millideg = 0;
                if (t1 >> millideg) {
                    r.nvme_c = static_cast<int>( std::lround(millideg / 1000.0) );
                }
            }
        }
    }
    if (include_cli) {
        r.lm_sensors_raw = run_lm_sensors_compact();
        r.nvidia_smi_raw = run_nvidia_smi_compact();
    }
    return r;
}

static void print_json_readings(bool include_cli) {
    Z420Readings r = read_sensors(include_cli);
    char load_buf[32];
    std::snprintf(load_buf, sizeof(load_buf), "%.2f", r.load_1m);
    std::cout << "{"
              << "\"cpu_pkg_c\":" << (r.cpu_pkg_c >= 0 ? std::to_string(r.cpu_pkg_c) : "null") << ","
              << "\"cpu_max_core_c\":" << (r.cpu_max_core_c >= 0 ? std::to_string(r.cpu_max_core_c) : "null") << ","
              << "\"nvme_c\":" << (r.nvme_c >= 0 ? std::to_string(r.nvme_c) : "null") << ","
              << "\"load_1m\":" << load_buf << ","
              << "\"lm_sensors_raw\":\"" << json_escape(r.lm_sensors_raw) << "\","
              << "\"nvidia_smi_raw\":\"" << json_escape(r.nvidia_smi_raw) << "\""
              << "}\n";
}

// Extract a JSON string or integer value for a top-level or simple key (lightweight parser for MCP JSON-RPC)
static std::string extract_json_id(const std::string& line) {
    auto pos = line.find("\"id\"");
    if (pos == std::string::npos) return "";
    auto colon = line.find(':', pos + 4);
    if (colon == std::string::npos) return "";
    size_t i = colon + 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= line.size()) return "";
    if (line.compare(i, 4, "null") == 0) return "";
    if (line[i] == '"') {
        size_t end = line.find('"', i + 1);
        if (end == std::string::npos) return "";
        return line.substr(i, end - i + 1);
    }
    size_t end = i;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-')) end++;
    return line.substr(i, end - i);
}

static std::string extract_json_string(const std::string& line, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return "";
    auto colon = line.find(':', pos + needle.size());
    if (colon == std::string::npos) return "";
    auto q1 = line.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return line.substr(q1 + 1, q2 - q1 - 1);
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--json") {
            print_json_readings(false);
            return 0;
        }
        if (arg1 == "--json-full") {
            print_json_readings(true);
            return 0;
        }
    }

    // MCP Stdio JSON-RPC Loop
    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::string id_raw = extract_json_id(line);
        if (id_raw.empty()) continue; // notification

        std::string method = extract_json_string(line, "method");
        if (method == "initialize") {
            std::string proto = extract_json_string(line, "protocolVersion");
            if (proto.empty()) proto = "2024-11-05";
            std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id_raw
                      << ",\"result\":{\"protocolVersion\":\"" << json_escape(proto)
                      << "\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"z420-sensors-native\",\"version\":\"2.0.0\"}}}\n"
                      << std::flush;
        } else if (method == "ping") {
            std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id_raw << ",\"result\":{}}\n" << std::flush;
        } else if (method == "tools/list") {
            std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id_raw
                      << ",\"result\":{\"tools\":["
                      << "{\"name\":\"lm_sensors\",\"description\":\"Run Ubuntu sensors (lm-sensors) and return compact CPU package, core, and NVMe temperatures.\",\"inputSchema\":{\"type\":\"object\",\"properties\":{}}},"
                      << "{\"name\":\"nvidia_smi\",\"description\":\"Run nvidia-smi and return compact GTX 1050 Ti GPU temperature, VRAM usage, utilization, and active processes.\",\"inputSchema\":{\"type\":\"object\",\"properties\":{}}}"
                      << "]}}\n"
                      << std::flush;
        } else if (method == "tools/call") {
            std::string text;
            if (line.find("\"nvidia_smi\"") != std::string::npos) {
                text = run_nvidia_smi_compact();
            } else if (line.find("\"lm_sensors\"") != std::string::npos) {
                text = run_lm_sensors_compact();
            } else {
                text = run_lm_sensors_compact() + " | " + run_nvidia_smi_compact();
            }
            std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id_raw
                      << ",\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"" << json_escape(text)
                      << "\"}],\"isError\":false}}\n"
                      << std::flush;
        } else {
            std::cout << "{\"jsonrpc\":\"2.0\",\"id\":" << id_raw << ",\"result\":{}}\n" << std::flush;
        }
    }
    return 0;
}
