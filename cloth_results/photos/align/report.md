# Alignment report: truth vs perception, prompt "fabric"

Per view: the renderer's truth (mask, metric depth, visible vertices) against the SAM 3 mask and
the VGGT cloud lifted inside it. Panels: `<set>__<stem>.png`. Worst first by the similarity RMSE;
views SAM did not find come first.

## cloth256: 48 views, SAM found 47, lifted 47

| metric (median over lifted views) | value |
|---|---|
| mask IoU | 0.986 |
| mask recall / precision | 0.990 / 0.999 |
| visible-vertex coverage | 1.000 |
| VGGT depth abs. error, median / p90 (cm) | 4.2 / 11.3 |
| VGGT depth bias (cm) | -0.8 |
| oracle depth range inside the mask (cm) | 44.7 |
| point RMSE, scale + centring / best similarity (cm) | 7.1 / 4.0 |
| fitted scale (VGGT -> metres) | 1.12 |

## paper256: 8 views, SAM found 8, lifted 8

| metric (median over lifted views) | value |
|---|---|
| mask IoU | 0.989 |
| mask recall / precision | 0.990 / 1.000 |
| visible-vertex coverage | 1.000 |
| VGGT depth abs. error, median / p90 (cm) | 3.7 / 11.3 |
| VGGT depth bias (cm) | -0.5 |
| oracle depth range inside the mask (cm) | 40.2 |
| point RMSE, scale + centring / best similarity (cm) | 7.3 / 4.9 |
| fitted scale (VGGT -> metres) | 1.11 |

## Per view (worst first)

| set | view | mat | visible | IoU | recall | precision | coverage | |dz| med cm | |dz| p90 cm | bias cm | depth range cm | RMSE scaled cm | RMSE sim cm | scale | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cloth256 | [sample_slot0001_seed100001_v3](cloth256__sample_slot0001_seed100001_v3.png) | cloth | 0.11 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  |  | SAM: no detection |
| cloth256 | [sample_slot0000_seed6000_v0](cloth256__sample_slot0000_seed6000_v0.png) | cloth | 0.19 | 0.984 | 0.987 | 0.997 | 1.000 | 7.2 | 25.2 | -3.3 | 67.3 | 14.3 | 12.8 | 1.11 |  |
| cloth256 | [sample_slot0002_seed100002_v1](cloth256__sample_slot0002_seed100002_v1.png) | cloth | 0.24 | 0.991 | 0.991 | 1.000 | 1.000 | 3.2 | 17.6 | -1.4 | 62.9 | 9.4 | 9.1 | 1.27 |  |
| cloth256 | [sample_slot0007_seed6007_v1](cloth256__sample_slot0007_seed6007_v1.png) | cloth | 0.13 | 0.943 | 0.990 | 0.952 | 1.000 | 3.5 | 17.4 | 0.1 | 58.9 | 9.0 | 8.5 | 0.91 |  |
| cloth256 | [sample_slot0005_seed6005_v2](cloth256__sample_slot0005_seed6005_v2.png) | cloth | 0.26 | 0.937 | 0.985 | 0.950 | 0.999 | 8.7 | 20.2 | -0.7 | 65.4 | 12.5 | 7.2 | 1.17 |  |
| cloth256 | [sample_slot0002_seed56004_v2](cloth256__sample_slot0002_seed56004_v2.png) | cloth | 0.27 | 0.985 | 0.986 | 0.999 | 1.000 | 7.9 | 17.6 | -0.8 | 45.2 | 11.2 | 7.2 | 1.12 |  |
| cloth256 | [sample_slot0007_seed6007_v0](cloth256__sample_slot0007_seed6007_v0.png) | cloth | 0.28 | 0.988 | 0.990 | 0.998 | 1.000 | 6.3 | 13.5 | -2.9 | 51.8 | 8.9 | 6.8 | 1.13 |  |
| cloth256 | [sample_slot0003_seed6003_v2](cloth256__sample_slot0003_seed6003_v2.png) | cloth | 0.45 | 0.982 | 0.991 | 0.990 | 1.000 | 9.4 | 23.4 | 3.4 | 28.9 | 13.1 | 6.4 | 1.18 |  |
| cloth256 | [sample_slot0001_seed6001_v1](cloth256__sample_slot0001_seed6001_v1.png) | cloth | 0.33 | 0.975 | 0.988 | 0.986 | 1.000 | 6.4 | 12.4 | -1.2 | 40.7 | 8.6 | 6.4 | 0.86 |  |
| cloth256 | [sample_slot0002_seed56004_v1](cloth256__sample_slot0002_seed56004_v1.png) | cloth | 0.28 | 0.986 | 0.987 | 0.999 | 1.000 | 9.2 | 18.7 | -1.8 | 51.8 | 12.0 | 6.3 | 1.08 |  |
| cloth256 | [sample_slot0003_seed6003_v1](cloth256__sample_slot0003_seed6003_v1.png) | cloth | 0.30 | 0.991 | 0.992 | 0.999 | 1.000 | 4.6 | 12.8 | -2.7 | 65.2 | 7.3 | 6.2 | 1.12 |  |
| paper256 | [sample_slot0001_seed100001_v3](paper256__sample_slot0001_seed100001_v3.png) | paper | 0.78 | 0.992 | 0.993 | 1.000 | 0.998 | 2.7 | 13.8 | -1.0 | 44.3 | 9.6 | 6.1 | 0.94 |  |
| cloth256 | [sample_slot0005_seed6005_v1](cloth256__sample_slot0005_seed6005_v1.png) | cloth | 0.20 | 0.884 | 0.984 | 0.896 | 0.998 | 7.1 | 16.0 | 0.1 | 68.3 | 10.2 | 6.1 | 0.85 |  |
| cloth256 | [sample_slot0006_seed56007_v1](cloth256__sample_slot0006_seed56007_v1.png) | cloth | 0.56 | 0.982 | 0.991 | 0.991 | 0.999 | 2.8 | 10.4 | -1.1 | 51.8 | 6.9 | 6.0 | 1.19 |  |
| cloth256 | [sample_slot0001_seed100001_v1](cloth256__sample_slot0001_seed100001_v1.png) | cloth | 0.30 | 0.990 | 0.990 | 1.000 | 0.999 | 7.9 | 16.2 | 1.7 | 77.6 | 10.3 | 6.0 | 0.92 |  |
| paper256 | [sample_slot0000_seed100000_v3](paper256__sample_slot0000_seed100000_v3.png) | paper | 0.58 | 0.992 | 0.992 | 1.000 | 1.000 | 5.8 | 15.9 | -1.8 | 23.7 | 10.2 | 5.7 | 1.08 |  |
| cloth256 | [sample_slot0002_seed100002_v3](cloth256__sample_slot0002_seed100002_v3.png) | cloth | 0.44 | 0.990 | 0.991 | 1.000 | 1.000 | 4.3 | 11.1 | -0.9 | 43.4 | 6.9 | 5.6 | 1.01 |  |
| cloth256 | [sample_slot0001_seed6001_v0](cloth256__sample_slot0001_seed6001_v0.png) | cloth | 0.17 | 0.986 | 0.987 | 0.999 | 1.000 | 6.4 | 15.6 | -3.5 | 36.5 | 7.8 | 5.6 | 1.11 |  |
| paper256 | [sample_slot0001_seed100001_v0](paper256__sample_slot0001_seed100001_v0.png) | paper | 0.51 | 0.988 | 0.988 | 1.000 | 1.000 | 3.5 | 11.5 | 1.4 | 41.6 | 8.6 | 5.6 | 1.33 |  |
| cloth256 | [sample_slot0002_seed56004_v3](cloth256__sample_slot0002_seed56004_v3.png) | cloth | 0.38 | 0.929 | 0.985 | 0.942 | 0.992 | 3.1 | 9.0 | 0.4 | 41.8 | 6.0 | 5.5 | 0.88 |  |
| cloth256 | [sample_slot0002_seed100002_v0](cloth256__sample_slot0002_seed100002_v0.png) | cloth | 0.38 | 0.992 | 0.993 | 1.000 | 1.000 | 9.1 | 14.1 | -0.3 | 57.0 | 11.4 | 5.4 | 1.12 |  |
| cloth256 | [sample_slot0006_seed56007_v2](cloth256__sample_slot0006_seed56007_v2.png) | cloth | 0.58 | 0.991 | 0.992 | 0.999 | 1.000 | 3.4 | 13.3 | -2.8 | 44.7 | 6.7 | 5.2 | 1.23 |  |
| paper256 | [sample_slot0000_seed100000_v1](paper256__sample_slot0000_seed100000_v1.png) | paper | 0.33 | 0.988 | 0.989 | 0.999 | 1.000 | 1.9 | 11.0 | -1.0 | 36.0 | 5.5 | 5.1 | 0.95 |  |
| cloth256 | [sample_slot0007_seed6007_v2](cloth256__sample_slot0007_seed6007_v2.png) | cloth | 0.35 | 0.988 | 0.989 | 0.999 | 1.000 | 6.8 | 11.3 | -0.7 | 64.2 | 8.2 | 4.9 | 1.28 |  |
| paper256 | [sample_slot0001_seed100001_v2](paper256__sample_slot0001_seed100001_v2.png) | paper | 0.36 | 0.984 | 0.985 | 1.000 | 1.000 | 5.5 | 7.3 | 3.5 | 38.9 | 6.1 | 4.7 | 1.32 |  |
| cloth256 | [sample_slot0001_seed100001_v0](cloth256__sample_slot0001_seed100001_v0.png) | cloth | 0.41 | 0.991 | 0.991 | 1.000 | 1.000 | 5.0 | 11.3 | 1.6 | 69.1 | 7.6 | 4.7 | 1.27 |  |
| cloth256 | [sample_slot0001_seed100001_v2](cloth256__sample_slot0001_seed100001_v2.png) | cloth | 0.26 | 0.980 | 0.981 | 0.999 | 0.994 | 1.8 | 16.9 | -0.8 | 54.1 | 8.0 | 4.4 | 0.94 |  |
| cloth256 | [sample_slot0004_seed56005_v2](cloth256__sample_slot0004_seed56005_v2.png) | cloth | 0.20 | 0.988 | 0.989 | 0.999 | 1.000 | 4.1 | 8.1 | -0.9 | 56.7 | 6.0 | 4.3 | 1.32 |  |
| cloth256 | [sample_slot0004_seed56005_v0](cloth256__sample_slot0004_seed56005_v0.png) | cloth | 0.41 | 0.987 | 0.991 | 0.996 | 0.999 | 4.4 | 10.5 | -1.9 | 43.9 | 6.3 | 4.2 | 1.02 |  |
| cloth256 | [sample_slot0000_seed6000_v3](cloth256__sample_slot0000_seed6000_v3.png) | cloth | 0.64 | 0.978 | 0.991 | 0.987 | 1.000 | 4.2 | 13.1 | -1.2 | 50.0 | 7.7 | 4.0 | 1.07 |  |
| paper256 | [sample_slot0000_seed100000_v2](paper256__sample_slot0000_seed100000_v2.png) | paper | 0.41 | 0.992 | 0.992 | 1.000 | 1.000 | 5.1 | 12.2 | -1.6 | 23.5 | 8.4 | 4.0 | 1.23 |  |
| cloth256 | [sample_slot0001_seed6001_v3](cloth256__sample_slot0001_seed6001_v3.png) | cloth | 0.22 | 0.983 | 0.990 | 0.993 | 1.000 | 4.6 | 11.4 | -0.4 | 29.4 | 7.3 | 4.0 | 1.06 |  |
| cloth256 | [sample_slot0003_seed100003_v1](cloth256__sample_slot0003_seed100003_v1.png) | cloth | 0.83 | 0.988 | 0.988 | 1.000 | 0.994 | 6.7 | 12.0 | -5.6 | 33.4 | 7.2 | 3.9 | 0.91 |  |
| cloth256 | [sample_slot0000_seed6000_v2](cloth256__sample_slot0000_seed6000_v2.png) | cloth | 0.19 | 0.984 | 0.986 | 0.998 | 1.000 | 3.3 | 7.7 | 0.2 | 62.1 | 5.0 | 3.9 | 1.20 |  |
| cloth256 | [sample_slot0002_seed100002_v2](cloth256__sample_slot0002_seed100002_v2.png) | cloth | 0.35 | 0.991 | 0.991 | 1.000 | 1.000 | 6.5 | 16.2 | -0.8 | 44.0 | 9.8 | 3.8 | 1.34 |  |
| cloth256 | [sample_slot0000_seed100000_v3](cloth256__sample_slot0000_seed100000_v3.png) | cloth | 0.33 | 0.987 | 0.988 | 1.000 | 1.000 | 2.2 | 7.1 | -1.3 | 39.2 | 4.6 | 3.7 | 1.01 |  |
| cloth256 | [sample_slot0003_seed100003_v2](cloth256__sample_slot0003_seed100003_v2.png) | cloth | 0.22 | 0.992 | 0.993 | 0.999 | 1.000 | 1.6 | 5.4 | -0.9 | 54.9 | 3.9 | 3.7 | 0.98 |  |
| cloth256 | [sample_slot0002_seed56004_v0](cloth256__sample_slot0002_seed56004_v0.png) | cloth | 0.27 | 0.956 | 0.981 | 0.974 | 0.999 | 2.7 | 5.4 | 0.7 | 38.6 | 3.8 | 3.6 | 1.00 |  |
| cloth256 | [sample_slot0003_seed6003_v3](cloth256__sample_slot0003_seed6003_v3.png) | cloth | 0.40 | 0.992 | 0.992 | 0.999 | 1.000 | 6.1 | 12.7 | 4.1 | 25.6 | 7.6 | 3.4 | 1.38 |  |
| cloth256 | [sample_slot0005_seed6005_v3](cloth256__sample_slot0005_seed6005_v3.png) | cloth | 0.41 | 0.933 | 0.989 | 0.942 | 0.996 | 4.8 | 11.7 | 0.8 | 60.5 | 7.1 | 3.4 | 1.12 |  |
| cloth256 | [sample_slot0000_seed6000_v1](cloth256__sample_slot0000_seed6000_v1.png) | cloth | 0.33 | 0.991 | 0.992 | 0.999 | 1.000 | 5.0 | 9.6 | 0.3 | 46.4 | 6.7 | 3.4 | 1.25 |  |
| cloth256 | [sample_slot0000_seed100000_v1](cloth256__sample_slot0000_seed100000_v1.png) | cloth | 0.31 | 0.987 | 0.989 | 0.997 | 1.000 | 2.0 | 6.4 | -1.6 | 45.3 | 4.5 | 3.3 | 1.22 |  |
| cloth256 | [sample_slot0003_seed100003_v0](cloth256__sample_slot0003_seed100003_v0.png) | cloth | 0.65 | 0.993 | 0.993 | 1.000 | 1.000 | 3.4 | 9.8 | -1.0 | 33.2 | 7.4 | 3.3 | 1.24 |  |
| cloth256 | [sample_slot0001_seed6001_v2](cloth256__sample_slot0001_seed6001_v2.png) | cloth | 0.39 | 0.984 | 0.985 | 0.999 | 1.000 | 2.0 | 9.8 | -0.2 | 29.5 | 5.4 | 3.1 | 1.21 |  |
| paper256 | [sample_slot0001_seed100001_v1](paper256__sample_slot0001_seed100001_v1.png) | paper | 0.34 | 0.987 | 0.987 | 1.000 | 1.000 | 3.8 | 6.7 | 3.3 | 44.7 | 4.7 | 3.1 | 1.09 |  |
| cloth256 | [sample_slot0004_seed56005_v1](cloth256__sample_slot0004_seed56005_v1.png) | cloth | 0.46 | 0.969 | 0.979 | 0.990 | 0.994 | 1.9 | 5.8 | 1.3 | 30.3 | 3.1 | 2.9 | 1.11 |  |
| cloth256 | [sample_slot0004_seed56005_v3](cloth256__sample_slot0004_seed56005_v3.png) | cloth | 0.52 | 0.961 | 0.989 | 0.971 | 1.000 | 3.1 | 7.7 | -1.2 | 27.7 | 4.6 | 2.8 | 1.09 |  |
| cloth256 | [sample_slot0000_seed100000_v0](cloth256__sample_slot0000_seed100000_v0.png) | cloth | 0.30 | 0.992 | 0.993 | 0.999 | 1.000 | 2.1 | 5.7 | -0.8 | 41.5 | 5.0 | 2.7 | 1.33 |  |
| cloth256 | [sample_slot0003_seed6003_v0](cloth256__sample_slot0003_seed6003_v0.png) | cloth | 0.50 | 0.987 | 0.990 | 0.996 | 1.000 | 2.3 | 5.9 | -0.8 | 28.9 | 3.7 | 2.7 | 1.17 |  |
| cloth256 | [sample_slot0000_seed100000_v2](cloth256__sample_slot0000_seed100000_v2.png) | cloth | 0.47 | 0.993 | 0.993 | 1.000 | 1.000 | 4.0 | 7.1 | -2.2 | 30.5 | 5.9 | 2.7 | 1.12 |  |
| cloth256 | [sample_slot0005_seed6005_v0](cloth256__sample_slot0005_seed6005_v0.png) | cloth | 0.27 | 0.962 | 0.985 | 0.976 | 0.988 | 5.2 | 8.3 | -1.5 | 28.9 | 6.1 | 2.3 | 1.18 |  |
| cloth256 | [sample_slot0007_seed6007_v3](cloth256__sample_slot0007_seed6007_v3.png) | cloth | 0.50 | 0.986 | 0.989 | 0.996 | 1.000 | 3.5 | 6.7 | -1.7 | 53.5 | 5.3 | 2.3 | 1.30 |  |
| cloth256 | [sample_slot0006_seed56007_v0](cloth256__sample_slot0006_seed56007_v0.png) | cloth | 0.72 | 0.971 | 0.993 | 0.977 | 1.000 | 1.1 | 5.7 | -0.5 | 34.3 | 3.1 | 2.2 | 1.06 |  |
| cloth256 | [sample_slot0006_seed56007_v3](cloth256__sample_slot0006_seed56007_v3.png) | cloth | 0.64 | 0.949 | 0.992 | 0.956 | 1.000 | 3.1 | 6.7 | 0.5 | 27.3 | 4.6 | 2.1 | 1.06 |  |
| paper256 | [sample_slot0000_seed100000_v0](paper256__sample_slot0000_seed100000_v0.png) | paper | 0.17 | 0.990 | 0.991 | 0.999 | 1.000 | 2.6 | 7.5 | 0.0 | 47.3 | 4.9 | 1.9 | 1.13 |  |
| cloth256 | [sample_slot0003_seed100003_v3](cloth256__sample_slot0003_seed100003_v3.png) | cloth | 0.29 | 0.987 | 0.988 | 0.999 | 1.000 | 1.5 | 2.9 | 0.1 | 34.2 | 2.9 | 1.2 | 0.78 |  |
