# Phân tích dự án StockSight

Tài liệu này dùng để học thi vấn đáp và thuyết trình cho đề tài **“Dự đoán giá cổ phiếu bằng mô hình học máy và học sâu”**. Nội dung được viết theo hướng dễ hiểu cho sinh viên mới học Machine Learning và Deep Learning.

## 1. Dự án StockSight làm gì?

StockSight là một hệ thống dự đoán giá và xu hướng cổ phiếu Mỹ. Dự án không chỉ train mô hình rồi in ra vài con số, mà xây dựng gần như một pipeline hoàn chỉnh:

1. Lấy dữ liệu giá cổ phiếu và chỉ số vĩ mô từ Yahoo Finance.
2. Lưu dữ liệu vào cache dạng parquet để chạy lại nhanh hơn.
3. Tạo các đặc trưng đầu vào, gọi là **features**.
4. Tạo nhãn cần dự đoán, gọi là **target**.
5. Chia dữ liệu thành train, validation và test theo thứ tự thời gian.
6. Train các mô hình Machine Learning để dự đoán lợi suất ngày tiếp theo.
7. Train mô hình Deep Learning để dự đoán chuỗi lợi suất 14 ngày tiếp theo.
8. Đánh giá mô hình bằng MAE, RMSE, directional accuracy, F1, MAPE.
9. Lưu dự đoán để backtest chiến lược giao dịch.
10. Hiển thị kết quả bằng giao diện Streamlit.

Điểm quan trọng: phần lớn mô hình trong dự án dự đoán **return**, tức tỷ lệ tăng/giảm, chứ không dự đoán trực tiếp giá tuyệt đối. Ví dụ giá hôm nay là 100 USD, mô hình dự đoán return là 0.02 thì giá dự đoán ngày mai là 102 USD.

## 2. Các thuật ngữ cơ bản

**Ticker**: Mã cổ phiếu, ví dụ `COST`, `BKR`, `AEP`.

**OHLCV**: Bộ dữ liệu giá gồm `open`, `high`, `low`, `close`, `volume`.

**Close price**: Giá đóng cửa cuối ngày. Dự án dùng giá đóng cửa làm mốc chính.

**Return**: Tỷ lệ thay đổi giá giữa hai ngày:

```text
return_t = close_t / close_(t-1) - 1
```

Nếu giá tăng từ 100 lên 103 thì return là 3%.

**Feature**: Biến đầu vào cho mô hình. Ví dụ: return ngày trước, RSI, MACD, volatility, VIX change.

**Target**: Giá trị mô hình cần học để dự đoán. Trong dự án:

```text
y_return = close_ngày_mai / close_hôm_nay - 1
```

**Machine Learning**: Nhóm mô hình học từ dữ liệu dạng bảng. Dự án dùng Ridge, Random Forest và XGBoost.

**Deep Learning**: Nhóm mô hình neural network nhiều lớp. Dự án dùng mô hình CNN + BiLSTM + Attention.

**Train set**: Dữ liệu quá khứ dùng để huấn luyện mô hình.

**Validation set**: Phần dữ liệu dùng để theo dõi chất lượng trong quá trình train hoặc báo cáo độ lệch giữa train và test.

**Test set**: Dữ liệu mới hơn, không dùng để train, chỉ dùng để đánh giá cuối cùng.

**Data leakage**: Lỗi khi thông tin tương lai bị lọt vào quá trình huấn luyện. Đây là lỗi rất nghiêm trọng trong bài toán chuỗi thời gian.

**Backtest**: Mô phỏng nếu dùng dự đoán của mô hình để giao dịch trong quá khứ thì kết quả lợi nhuận và rủi ro ra sao.

## 3. Cấu trúc thư mục

```text
stocksight/
  config.yaml                  Cấu hình ticker, macro, split, hyperparameter
  requirements.txt             Danh sách thư viện Python cần cài
  README.md                    Mô tả ngắn về dự án
  STOCKSIGHT_ANALYSIS.md       Tài liệu phân tích này

  data/
    raw/stocks/*.parquet       Cache dữ liệu giá cổ phiếu
    raw/macro/*.parquet        Cache dữ liệu chỉ số vĩ mô

  models/                      Model và scaler đã train
    {TICKER}_ridge.pkl
    {TICKER}_rf_reg.pkl
    {TICKER}_xgb_reg.pkl
    {TICKER}_dl.keras
    {TICKER}_scaler_X.joblib

  results/                     Kết quả train, prediction, backtest, hình ảnh
    ml_summary.csv
    dl_summary.csv
    predictions_ml.parquet
    predictions_dl.parquet
    backtest_summary.csv

  src/                         Source code xử lý dữ liệu, model, evaluate
  scripts/                     Các script chạy pipeline
  streamlit_app/               Giao diện web Streamlit
  tests/                       Unit test kiểm tra logic quan trọng
```

## 4. Công nghệ sử dụng

**pandas, numpy**: Xử lý bảng dữ liệu và tính toán số học.

**yfinance**: Tải dữ liệu tài chính từ Yahoo Finance.

**pyarrow/parquet**: Lưu dữ liệu cache dạng parquet.

**scikit-learn**: Dùng `MinMaxScaler`, Ridge, Random Forest và các metric.

**xgboost**: Dùng mô hình `XGBRegressor`.

**tensorflow/keras**: Xây dựng và train mô hình Deep Learning.

**joblib**: Lưu và đọc lại model `.pkl`, scaler `.joblib`.

**matplotlib, plotly, seaborn**: Vẽ biểu đồ.

**streamlit**: Xây dựng giao diện dashboard.

**shap**: Giải thích mô hình tree-based, xem feature nào ảnh hưởng đến prediction.

## 5. Cấu hình chính trong `config.yaml`

`data.start_date = "1990-01-01"`: Lấy dữ liệu từ năm 1990.

`tickers`: Danh sách 21 mã cổ phiếu Mỹ.

`macro_indices` gồm:

- `^VIX`: Chỉ số biến động thị trường.
- `^TNX`: Lợi suất trái phiếu Mỹ kỳ hạn 10 năm.
- `CL=F`: Giá dầu thô.
- `DX-Y.NYB`: Chỉ số đồng USD.

`split.train_ratio = 0.85`: 85% dữ liệu đầu tiên dùng cho train/validation, 15% cuối dùng cho test.

`dl.window = 60`: Mô hình Deep Learning nhìn 60 ngày quá khứ.

`dl.horizon = 14`: Mô hình Deep Learning dự đoán 14 ngày tương lai.

`backtest.threshold = 0.005`: Ngưỡng tín hiệu 0.5% để quyết định giao dịch.

## 6. Luồng xử lý tổng quát

```text
config.yaml
   |
   v
src/data_loader.py
   Tải dữ liệu stock và macro từ yfinance, cache parquet
   |
   v
src/preprocessing.py -> build_macro_df()
   Tạo macro_df: vix_change, tnx_change, oil_change, usd_change
   |
   v
src/features.py -> compute_features()
   Tạo returns, lag, MA, EMA, RSI, MACD, volatility, Bollinger,
   volume_ratio, macro feature, sector correlation
   |
   v
src/targets.py -> make_targets()
   Tạo y_return = return ngày mai
   |
   v
src/preprocessing.py
   prepare_ml_split(): dữ liệu dạng bảng cho ML
   build_dl_sequences(): chuỗi 60 ngày -> target 14 ngày cho DL
   |
   v
src/models/ml_model.py và src/models/dl_model.py
   Train model
   |
   v
src/evaluation_report.py
   Tính metric và lưu prediction
   |
   v
scripts/train_all.py
   Chạy toàn bộ 21 ticker
   |
   v
src/backtest.py và scripts/run_backtest.py
   Mô phỏng giao dịch
   |
   v
streamlit_app/
   Hiển thị dashboard, forecast, compare, backtest, SHAP
```

## 7. Giải thích các file quan trọng

### `src/config.py`

File này đọc cấu hình từ `config.yaml`, tạo các đường dẫn gốc như `DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR` và tự tạo thư mục nếu chưa tồn tại. Đây là file giúp các module khác không phải tự ghi cứng đường dẫn.

### `src/data_loader.py`

File này chịu trách nhiệm lấy dữ liệu từ Yahoo Finance.

Hàm `_fetch(symbol, start)`:

1. Gọi `yf.download()` để tải dữ liệu.
2. Dùng `auto_adjust=True` để giá được điều chỉnh theo chia tách cổ phiếu/cổ tức.
3. Đưa `Date` từ index thành cột `date`.
4. Chuẩn hóa tên cột về chữ thường.
5. Trả về dataframe có các cột như `date`, `open`, `high`, `low`, `close`, `volume`.

Hàm `load_stock(ticker, refresh=False)`:

1. Kiểm tra file cache `data/raw/stocks/{ticker}.parquet`.
2. Nếu có cache và không yêu cầu refresh thì đọc từ cache.
3. Nếu chưa có thì tải từ Yahoo Finance và lưu cache.

Hàm `load_macro(symbol, refresh=False)` làm tương tự cho các chỉ số vĩ mô.

### `src/sector_config.py`

File này gán mỗi ticker vào một ngành. Ví dụ:

- `BKR`, `APA`: năng lượng.
- `FITB`, `NTRS`: tài chính.
- `AEP`: tiện ích.

Mục đích là tạo feature theo ngành:

- Nhóm nhạy với dầu sẽ có feature tương quan với dầu.
- Nhóm nhạy với lãi suất sẽ có feature tương quan với lãi suất.

### `src/features.py`

Đây là file feature engineering.

Các nhóm feature chính:

- `returns`: lợi suất ngày.
- `log_returns`: lợi suất log.
- `close_pct_lag_1`, `close_pct_lag_5`: biến động so với 1 ngày và 5 ngày trước.
- `ma_5`, `ma_20`: trung bình trượt.
- `ema_12`, `ema_26`: trung bình mũ.
- `rsi_14`: chỉ báo sức mạnh tương đối.
- `macd`, `macd_signal`: chỉ báo xu hướng/động lượng.
- `volatility_21`: độ biến động 21 ngày.
- `bb_width`: độ rộng Bollinger Band.
- `volume_ratio`: volume hôm nay so với volume trung bình 20 ngày.
- `vix_change`, `tnx_change`, `oil_change`, `usd_change`: biến động chỉ số vĩ mô.
- `stock_oil_corr_21`, `vol_x_oil`: feature cho nhóm nhạy với dầu.
- `stock_rate_corr_21`: feature cho nhóm nhạy với lãi suất.

Hàm `compute_features()` tạo tất cả feature theo thứ tự. Các feature rolling chỉ dùng dữ liệu hiện tại và quá khứ, không dùng dữ liệu tương lai.

Hàm `feature_columns(ticker)` trả về danh sách feature cuối cùng cho từng ticker. Cột `close` không đưa vào input vì model học return, còn `close` chỉ dùng làm mốc quy đổi return thành giá.

### `src/targets.py`

File này tạo target cho Machine Learning:

```python
df["y_return"] = df["close"].shift(-1) / df["close"] - 1
```

Tại một ngày `t`, feature là thông tin đến ngày `t`, còn target là return của ngày `t+1`.

Ví dụ:

```text
close hôm nay = 100
close ngày mai = 102
y_return = 102 / 100 - 1 = 0.02
```

Dòng cuối cùng không có ngày mai nên target là `NaN`, sau đó pipeline sẽ loại bỏ bằng `dropna()`.

### `src/preprocessing.py`

Đây là file rất quan trọng vì xử lý dữ liệu trước khi đưa vào model.

`build_macro_df()`:

1. Tải các chỉ số vĩ mô.
2. Tính phần trăm thay đổi của từng chỉ số.
3. Gộp thành một bảng có các cột `vix_change`, `tnx_change`, `oil_change`, `usd_change`.

`prepare_ml_split()`:

1. Sắp xếp dữ liệu theo ngày.
2. Tách `X` là feature và `y_return` là target.
3. Chia dữ liệu theo thời gian, không random.
4. Fit `MinMaxScaler` chỉ trên train.
5. Transform test bằng scaler đã học từ train.
6. Cắt 10% cuối của train làm validation.

Lý do không random split: dữ liệu cổ phiếu là chuỗi thời gian. Nếu random, mô hình có thể học từ dữ liệu tương lai để dự đoán quá khứ, gây sai lệch nghiêm trọng.

`build_dl_sequences()` tạo dữ liệu chuỗi cho Deep Learning.

Với `window=60`, `horizon=14`:

```text
Input X: feature của ngày t-59 đến ngày t
Target y: return của ngày t+1 đến ngày t+14
```

Ví dụ:

```text
sample 1: ngày 1-60  -> dự đoán ngày 61-74
sample 2: ngày 2-61  -> dự đoán ngày 62-75
sample 3: ngày 3-62  -> dự đoán ngày 63-76
```

Target của DL là raw return, không scale, để loss function còn biết return âm hay dương.

### `src/models/ml_model.py`

File này train ba mô hình Machine Learning:

**Ridge Regression**: Hồi quy tuyến tính có regularization L2. Mô hình đơn giản, ít overfit hơn hồi quy tuyến tính thường.

**Random Forest Regressor**: Tập hợp nhiều cây quyết định. Kết quả là trung bình của nhiều cây, giúp bắt quan hệ phi tuyến.

**XGBoost Regressor**: Mô hình boosting, trong đó cây sau học cách sửa lỗi của cây trước. Đây là mô hình mạnh cho dữ liệu dạng bảng.

Mỗi hàm train trả về:

```text
model, y_pred_test, y_pred_val
```

Hàm `recursive_forecast_14()` dùng khi ML cần dự đoán nhiều ngày. Vì ML chỉ dự đoán 1 ngày, nên muốn dự đoán 14 ngày phải:

1. Dự đoán return ngày 1.
2. Cập nhật các feature lag bằng return vừa dự đoán.
3. Dự đoán ngày 2.
4. Lặp lại đến ngày 14.

### `src/models/dl_model.py`

Đây là file xây dựng mô hình Deep Learning.

Kiến trúc:

```text
Input: (60 ngày, số feature)
  -> Conv1D
  -> BatchNormalization
  -> Dropout
  -> Conv1D
  -> Bidirectional LSTM encoder
  -> lấy trạng thái cuối
  -> RepeatVector(14)
  -> LSTM decoder
  -> Attention
  -> Dense theo từng ngày
  -> Output: (14 ngày, 1 return)
```

Giải thích:

**Conv1D** phát hiện các pattern ngắn hạn trong chuỗi 60 ngày.

**BatchNormalization** giúp quá trình train ổn định hơn.

**Dropout** giảm overfitting bằng cách tắt ngẫu nhiên một phần neuron khi train.

**BiLSTM** đọc chuỗi trong cửa sổ lịch sử và mã hóa thông tin quá khứ.

**RepeatVector(14)** nhân bản vector tóm tắt thành 14 bước để decoder tạo 14 dự đoán.

**Attention** giúp decoder nhìn lại toàn bộ chuỗi encoder và tập trung vào những ngày quan trọng hơn.

Loss của mô hình là `directional_loss`. Đây là MSE nhưng nếu mô hình dự đoán sai hướng tăng/giảm thì lỗi bị nhân thêm penalty. Điều này hợp lý trong bài toán cổ phiếu vì dự đoán đúng hướng tăng/giảm rất quan trọng.

Output dùng `tanh * return_cap`. Với `return_cap=0.05`, return dự đoán mỗi ngày bị giới hạn khoảng từ -5% đến +5%, tránh dự đoán quá cực đoan.

### `src/evaluation_report.py`

File này tính metric đánh giá.

Với ML:

- `test_mae_return`: sai số tuyệt đối trung bình của return.
- `test_rmse_return`: RMSE của return.
- `mae_vs_naive_ratio`: so với baseline dự đoán return bằng 0.
- `r2_score`: mức giải thích phương sai target.
- `directional_accuracy`: tỷ lệ dự đoán đúng hướng tăng/giảm.
- `residual_std`, `residual_skew`, `residual_kurt`: thống kê phần dư.

Với DL:

1. Dự đoán chuỗi return 14 ngày.
2. Quy đổi return thành giá bằng giá anchor.
3. Tính MAE, RMSE, MAPE theo giá.
4. Tính directional accuracy theo từng ngày và theo cả 14 ngày.

### `src/baselines.py`

Baseline là mô hình rất đơn giản để làm mốc so sánh:

- `naive_zero`: dự đoán ngày mai return = 0.
- `naive_persistence`: dự đoán ngày mai giống return hôm nay.

Nếu model phức tạp không tốt hơn baseline thì model chưa chứng minh được giá trị.

### `src/backtest.py`

Backtest mô phỏng chiến lược giao dịch từ prediction.

Chiến lược chính:

1. Mỗi ngày có signal dự báo cho từng ticker.
2. Chọn các ticker có signal lớn hơn threshold.
3. Lấy tối đa `top_k` ticker có signal cao nhất.
4. Gán trọng số bằng nhau `1/top_k`.
5. Phần còn lại là tiền mặt nếu không đủ ticker đạt ngưỡng.

Dòng quan trọng:

```python
positions = weights_today.shift(1).fillna(0.0)
```

Nghĩa là tín hiệu ngày `t` chỉ được dùng để nắm giữ vị thế từ ngày `t+1`. Đây là cách tránh look-ahead bias.

Metric backtest:

- `total_return`: lợi nhuận tổng.
- `ann_return`: lợi nhuận năm hóa.
- `ann_vol`: biến động năm hóa.
- `sharpe`: lợi nhuận trên rủi ro.
- `max_dd`: mức sụt giảm lớn nhất từ đỉnh.
- `hit_rate`: tỷ lệ ngày có lãi.
- `avg_turnover`: mức thay đổi vị thế trung bình.
- `alpha_vs_bench`: alpha so với benchmark.

### `scripts/train_all.py`

Đây là script train chính cho toàn bộ 21 ticker.

Với mỗi ticker:

1. Chuẩn bị dữ liệu.
2. Train Ridge, Random Forest, XGBoost.
3. Lưu model ML và scaler.
4. Tính metric ML và baseline.
5. Lưu prediction ML.
6. Train mô hình Deep Learning.
7. Lưu model DL, history và scaler.
8. Tính metric DL.
9. Lưu prediction DL.
10. Ghi summary CSV.

Output chính:

- `results/ml_summary.csv`
- `results/dl_summary.csv`
- `results/predictions_ml.parquet`
- `results/predictions_dl.parquet`
- `results/run_metadata.json`

### `streamlit_app/`

Giao diện Streamlit gồm các trang:

**Home.py**: Hiển thị tổng quan 21 ticker, sparkline, candlestick, RSI/MACD, Bollinger, correlation.

**1_Predict.py**: Cho người dùng chọn ticker, model, horizon rồi chạy dự báo live.

**2_Compare_Backtest.py**: So sánh model và chạy backtest cho một mã cổ phiếu.

**3_Explain.py**: Dùng SHAP để giải thích Random Forest và XGBoost.

## 8. Machine Learning và Deep Learning khác nhau thế nào trong dự án?

Machine Learning:

```text
Input: 1 dòng feature của một ngày
Output: return ngày mai
```

Deep Learning:

```text
Input: chuỗi feature 60 ngày
Output: chuỗi return 14 ngày tương lai
```

ML đơn giản hơn, train nhanh hơn, dễ giải thích hơn. DL phức tạp hơn, train lâu hơn nhưng có khả năng học pattern theo chuỗi thời gian tốt hơn.

## 9. Đánh giá mô hình

**MAE**:

```text
MAE = trung bình |giá trị thật - giá trị dự đoán|
```

MAE càng nhỏ càng tốt.

**RMSE**:

```text
RMSE = sqrt(trung bình sai số bình phương)
```

RMSE phạt nặng các lỗi lớn.

**Directional accuracy**:

```text
sign(y_pred) == sign(y_true)
```

Nghĩa là mô hình dự đoán đúng hướng tăng hoặc giảm.

**MAPE**:

```text
MAPE = trung bình |giá thật - giá dự đoán| / giá thật
```

MAPE cho biết sai số theo phần trăm.

**F1**: Chỉ số cân bằng giữa precision và recall khi xem dự đoán tăng giá như một bài toán phân loại.

## 10. Backtest giải thích đơn giản

Backtest trả lời câu hỏi: nếu dùng prediction của model để giao dịch trong quá khứ thì có hiệu quả không?

Ví dụ model dự đoán:

```text
COST: +1.2%
AEP: +0.7%
BKR: -0.4%
```

Nếu threshold là 0.5%, hệ thống sẽ chọn COST và AEP, không chọn BKR. Nếu `top_k=5`, mỗi mã được 20% vốn, phần còn lại giữ tiền mặt.

Backtest có tính chi phí giao dịch:

```text
cost = cost_per_trade * tổng thay đổi trọng số danh mục
```

## 11. Cách chạy dự án

Cài thư viện:

```bash
pip install -r requirements.txt
```

Tải dữ liệu:

```bash
python -m scripts.prepare_data
```

Chạy kiểm tra nhanh:

```bash
python -m scripts.smoke_test
```

Train toàn bộ model:

```bash
python -m scripts.train_all
```

In bảng summary:

```bash
python -m scripts.print_summary full
```

Chạy backtest:

```bash
python -m scripts.run_backtest
```

Chạy giao diện Streamlit:

```bash
streamlit run streamlit_app/Home.py
```

Thứ tự nên chạy khi demo:

```text
prepare_data -> smoke_test -> train_all -> run_backtest -> streamlit
```

Nếu thư mục `models/` và `results/` đã có sẵn kết quả thì có thể chạy Streamlit ngay.

## 12. Điểm mạnh của dự án

1. Có pipeline hoàn chỉnh từ dữ liệu đến dashboard.
2. Có cả Machine Learning và Deep Learning.
3. Có split theo thời gian, phù hợp với dữ liệu chứng khoán.
4. Có scaler fit trên train để tránh data leakage.
5. Có baseline để so sánh.
6. Có backtest, không chỉ đánh giá bằng metric ML.
7. Có SHAP để giải thích mô hình.
8. Có cache parquet giúp chạy lại nhanh hơn.

## 13. Hạn chế của dự án

1. Dữ liệu chủ yếu lấy từ Yahoo Finance, chưa có tin tức, báo cáo tài chính, sentiment.
2. ML recursive forecast chỉ cập nhật một số lag feature, chưa tính lại toàn bộ RSI/MACD/MA cho tương lai.
3. Thị trường tài chính rất nhiễu nên metric có thể không cao.
4. Backtest dùng giả định đơn giản về khớp lệnh và chi phí giao dịch.
5. Model train riêng từng ticker, chưa học chung quan hệ giữa nhiều cổ phiếu.
6. Hyperparameter chưa có quy trình tuning lớn.

Khi vấn đáp, nên nói rằng dự án tập trung xây dựng pipeline học máy hoàn chỉnh và kiểm soát data leakage. Hướng phát triển tiếp theo là thêm dữ liệu tin tức, dữ liệu cơ bản doanh nghiệp, walk-forward validation và quản trị rủi ro.

## 14. Câu hỏi vấn đáp thường gặp

### 1. Đề tài của em giải quyết bài toán gì?

Dự án giải quyết bài toán dự đoán lợi suất và xu hướng giá cổ phiếu Mỹ. Machine Learning dự đoán return ngày tiếp theo, còn Deep Learning dự đoán chuỗi return 14 ngày. Kết quả được đánh giá bằng metric dự đoán và backtest giao dịch.

### 2. Vì sao dự đoán return thay vì giá?

Giá tuyệt đối của các cổ phiếu có thang đo khác nhau và thường không ổn định. Return thể hiện tỷ lệ thay đổi, dễ so sánh hơn và phù hợp hơn cho mô hình. Sau khi dự đoán return, ta có thể quy đổi lại thành giá bằng giá đóng cửa hiện tại.

### 3. Dữ liệu lấy từ đâu?

Dữ liệu lấy từ Yahoo Finance bằng thư viện `yfinance`. Dự án lấy giá OHLCV của 21 ticker và các chỉ số vĩ mô như VIX, lợi suất 10 năm, dầu thô, chỉ số USD.

### 4. Feature engineering gồm những gì?

Feature gồm ba nhóm: kỹ thuật, vĩ mô và theo ngành. Nhóm kỹ thuật có return, moving average, RSI, MACD, volatility, Bollinger width, volume ratio. Nhóm vĩ mô có VIX, lãi suất, dầu, USD. Nhóm theo ngành có tương quan với dầu hoặc lãi suất.

### 5. Dữ liệu được chia train/test như thế nào?

Dữ liệu được chia theo thứ tự thời gian. 85% đầu dùng cho train/validation, 15% cuối dùng cho test. Trong phần train, 10% cuối làm validation. Không dùng random split vì sẽ gây sai logic thời gian.

### 6. Data leakage là gì?

Data leakage là khi thông tin tương lai bị đưa vào quá trình train. Dự án tránh bằng cách split theo thời gian, fit scaler chỉ trên train và trong backtest dùng tín hiệu ngày hôm trước cho vị thế ngày hôm sau.

### 7. Mô hình ML nào được sử dụng?

Dự án dùng Ridge, Random Forest Regressor và XGBoost Regressor. Ridge là hồi quy tuyến tính có regularization, Random Forest là tập hợp nhiều cây quyết định, XGBoost là boosting mạnh cho dữ liệu dạng bảng.

### 8. Mô hình Deep Learning là gì?

Mô hình Deep Learning là CNN + BiLSTM + Attention. Input là 60 ngày feature gần nhất, output là 14 return tương lai. CNN bắt pattern ngắn hạn, BiLSTM học chuỗi thời gian, Attention giúp mô hình tập trung vào các ngày quan trọng trong quá khứ.

### 9. Directional loss là gì?

Directional loss là MSE nhưng nếu mô hình dự đoán sai hướng tăng/giảm thì lỗi bị nhân thêm penalty. Điều này quan trọng vì trong giao dịch, dự đoán đúng hướng tăng/giảm có ý nghĩa lớn.

### 10. Backtest hoạt động như thế nào?

Backtest đọc prediction, tạo tín hiệu, chọn top-k ticker có tín hiệu cao hơn threshold, gán trọng số bằng nhau, lùi tín hiệu một ngày để tránh nhìn trước tương lai, sau đó tính lợi nhuận danh mục sau chi phí giao dịch.

### 11. SHAP dùng để làm gì?

SHAP dùng để giải thích mô hình Random Forest và XGBoost. Nó cho biết feature nào làm prediction tăng hoặc giảm, và mức ảnh hưởng của từng feature.

### 12. Nếu demo dự án, em sẽ demo gì?

Em sẽ demo trang Home để xem tổng quan universe, trang Predict để dự báo 14 ngày, trang Compare & Backtest để so sánh model và mô phỏng giao dịch, cuối cùng là trang Explain để xem SHAP giải thích mô hình.

## 15. Tóm tắt thuyết trình ngắn

StockSight là hệ thống dự đoán giá cổ phiếu Mỹ bằng Machine Learning và Deep Learning. Dữ liệu được lấy từ Yahoo Finance, gồm giá OHLCV của 21 cổ phiếu và các chỉ số vĩ mô như VIX, lãi suất, dầu, USD. Dự án tạo feature kỹ thuật và vĩ mô, tạo target là return ngày tiếp theo, chia dữ liệu theo thời gian để tránh data leakage. Ba mô hình ML gồm Ridge, Random Forest và XGBoost dự đoán return một ngày. Mô hình Deep Learning CNN + BiLSTM + Attention nhận 60 ngày lịch sử và dự đoán 14 ngày return. Kết quả được đánh giá bằng MAE, RMSE, directional accuracy, MAPE và được kiểm tra thêm bằng backtest top-k có tính chi phí giao dịch. Giao diện Streamlit hiển thị dashboard, dự báo, so sánh mô hình, backtest và SHAP explainability.
