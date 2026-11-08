# 森林火场空间异质性及驱动机制 —— 自动化流水线

用火前因子（植被、地形、土地覆盖、气象）预测火场内部的连续 dNBR，并分析各类因子的贡献。

## 分工

| 环节 | 在哪里做 | 说明 |
|---|---|---|
| 遥感与气象数据 | 公开数据源，直接读取 | Sentinel-2 L2A（Element 84 Earth Search / AWS）、Copernicus DEM、ESA WorldCover、Open-Meteo ERA5-Land。都不需要账号 |
| 计算 | GitHub Actions | 每天自动处理新加入的火场；训练和测试手动触发 |
| 存储与记录 | Supabase 项目 `fire-research` | 火场名单、数据登记、实验记录、逐火场成绩、运行日志、文件 |

## 数据库表

- `fires`：火场名单。`split` 取值 dev（开发）或 test（封存测试）；锁定后数据库不允许再改。
- `data_assets`：每个火场每一层数据的位置、来源、版本、校验码。
- `experiments` / `fire_metrics`：每次实验的设置，以及**逐火场**的 R²、RMSE、MAE、偏差。
- `pipeline_runs`：每次自动运行的状态和日志。

## 统一的数据口径（v1，见 `config.yaml`）

- 火前合成：起火前 60 天到前 1 天；火后合成：结束后 3–45 天。用 SCL 逐像元去云，取中位数合成。
- 因变量：`dnbr_offset` = dNBR 减去火场外环（300–1500 m）未燃区的中位数，用来消除季节和物候带来的偏移。
- 火烧范围：有官方边界就用边界；没有就按 dNBR > 0.10（不取绝对值）自动提取，并标记为需要人工核对。
- 火后影像只用来生成标签，不作为预测因子。
- 气象约 10 km 一格，只反映火场之间的差异。火灾期间的天气属于“完美预报”情景，论文里需要写明。

## 验收口径

- 封存测试火场（约 20%，按省份分层随机抽取，锁定后不可改）逐场计算 R²，导师标准为 ≥ 0.90。
- 开发阶段只看开发火场的“按火场分组交叉验证”成绩。
- 最终测试每运行一次都会留下记录；第二次及以后的运行会自动注明“测试集已不再是全新的”。

## 自动发现火灾（VIIRS 火点）

GitHub → Actions →「用火点发现火灾」→ 填范围和日期 → 运行。程序会把 VIIRS 375 m 火点按时空聚类（相距 ≤1.5 km、相隔 ≤2 天算同一场），
第一个和最后一个火点的日期就是起止日期，火点中心就是坐标，然后写进名单，第二天凌晨自动处理。
已在名单里的火场（3 km、5 天内）不会重复加入。

## 常用命令

```bash
python -m firepipe.cli discover --bbox 111,24.6,114.3,27.5 --start 2022-09-01 --end 2022-11-30 --province 湖南
python -m firepipe.cli refine-dates                       # 用火点给缺日期的火场补起止日期
python -m firepipe.cli import-fires data/fires_seed.csv   # 导入火场
python -m firepipe.cli split --lock                       # 划分并锁定封存测试火场
python -m firepipe.cli process --limit 20                 # 处理待处理火场
python -m firepipe.cli train                              # 开发集交叉验证
python -m firepipe.cli ablation --algo lightgbm           # 删减因子组实验
python -m firepipe.cli final-test --algo lightgbm --confirm
```

## 需要在 GitHub 仓库 Settings → Secrets → Actions 里设置

- `SUPABASE_URL` = `https://cgoajvzmghvvxacimcvn.supabase.co`
- `SUPABASE_SERVICE_KEY` = Supabase 后台 → Project Settings → API Keys 里的 secret key（不要发到聊天里）
- `FIRMS_MAP_KEY` = NASA FIRMS 免费密钥（https://firms.modaps.eosdis.nasa.gov/api/map_key/ 填邮箱即可获得）

## 已知限制 / 下一步

- TWI 用的是没有填洼的简单 D8 汇流，属于近似。
- 冠层高度、森林类型、逐像元“燃烧当天”的天气，还没有接入（计划在 v2 加入；火点数据已经接入，用于确定起止日期）。
- 远程取数部分还没有在真实网络上跑过，第一次处理火场时需要查看运行日志。
