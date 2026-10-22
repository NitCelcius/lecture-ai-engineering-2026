| model | n_ok | n_error | latency_median | latency_min | latency_max | ttft_median | output_s_median | prompt_tokens | output_tokens | thinking_tokens | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 8 | 1 | 7.3 | 4.5 | 42.3 | 4.2 | 2.2 | 35.5 | 754.5 | 0 | 290.7 |
| gemini-3.5-flash-lite | 9 | 0 | 4.0 | 2.7 | 14.8 | 0.6 | 3.4 | 32.3 | 994.9 | 0 | 242.2 |
| gemini-3.8-flash | 6 | 3 | 27.5 | 20.1 | 55.6 | 25.8 | 5.2 | 22 | 813 | 1133.8 | 338.6 |
| gemma-4-31b-it | 6 | 3 | 76.3 | 49.3 | 105.5 | 49.7 | 16.4 | 34.7 | 837.8 | 682.7 | 37.0 |

| model | question_id | n_ok | n_error | latency_median | latency_min | latency_max | ttft_median | output_s_median | prompt_tokens | output_tokens | thinking_tokens | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | false_premise | 3 | 0 | 6.1 | 4.5 | 42.3 | 4.2 | 1.9 | 21 | 496.7 | 0 | 277.9 |
| gemini-3.1-flash-lite | long | 3 | 0 | 7.6 | 7.0 | 8.2 | 3.7 | 3.5 | 69 | 1068.7 | 0 | 296.6 |
| gemini-3.1-flash-lite | short | 2 | 1 | 7.5 | 6.3 | 8.7 | 5.2 | 2.2 | 7 | 670 | 0 | 301.0 |
| gemini-3.5-flash-lite | false_premise | 3 | 0 | 4.0 | 3.8 | 4.1 | 0.5 | 3.4 | 21 | 750 | 0 | 217.1 |
| gemini-3.5-flash-lite | long | 3 | 0 | 8.3 | 6.8 | 14.8 | 0.8 | 6.3 | 69 | 1691.3 | 0 | 255.5 |
| gemini-3.5-flash-lite | short | 3 | 0 | 2.7 | 2.7 | 3.0 | 0.6 | 2.1 | 7 | 543.3 | 0 | 254.1 |
| gemini-3.8-flash | false_premise | 2 | 1 | 26.8 | 26.6 | 27.1 | 24.9 | 1.9 | 21 | 204 | 1333 | 110.4 |
| gemini-3.8-flash | long | 1 | 2 | 27.9 | 27.9 | 27.9 | 26.6 | 1.3 | 69 | 1997 | 1142 | 1520.9 |
| gemini-3.8-flash | short | 3 | 0 | 39.6 | 20.1 | 55.6 | 31.3 | 8.3 | 7 | 824.3 | 998.3 | 96.5 |
| gemma-4-31b-it | false_premise | 3 | 0 | 51.2 | 49.3 | 63.4 | 38.9 | 11.2 | 21 | 405 | 684.7 | 37.5 |
| gemma-4-31b-it | long | 2 | 1 | 92.6 | 89.1 | 96.1 | 50.5 | 42.1 | 69 | 1530.5 | 720.5 | 36.3 |
| gemma-4-31b-it | short | 1 | 2 | 105.5 | 105.5 | 105.5 | 85.1 | 20.4 | 7 | 751 | 601 | 36.8 |

