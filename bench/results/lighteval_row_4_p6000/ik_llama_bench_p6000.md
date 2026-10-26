| model                          |       size |     params | backend    | ngl | type_k | type_v | mmap |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | ---: | ------------: | ---------------: |
| qwen35moe 35B.A3B Q6_K         |  17.90 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |    0 |         pp512 |   970.17 ± 49.30 |
| qwen35moe 35B.A3B Q6_K         |  17.90 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |    0 |         tg128 |     64.35 ± 0.27 |
| qwen35moe 35B.A3B Q6_K         |  21.24 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |    0 |         pp512 |   900.72 ± 17.22 |
| qwen35moe 35B.A3B Q6_K         |  21.24 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |    0 |         tg128 |     61.42 ± 0.20 |

build: 9bb70bc (1)
