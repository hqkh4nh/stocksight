# StockSight - Câu hỏi vấn đáp và trả lời mẫu

Tài liệu này tổng hợp các câu hỏi thường gặp khi bảo vệ đề tài **“Dự đoán giá cổ phiếu bằng mô hình học máy và học sâu”**. Câu trả lời được viết theo hướng ngắn gọn, dễ nói trong vấn đáp, nhưng vẫn bám sát source code của dự án StockSight.

## 1. Tổng quan đề tài

### 1.1 Đề tài của em giải quyết bài toán gì?

Dạ, đề tài của em xây dựng hệ thống dự đoán giá và xu hướng cổ phiếu Mỹ bằng Machine Learning và Deep Learning. Cụ thể, các mô hình Machine Learning dự đoán lợi suất ngày tiếp theo, còn mô hình Deep Learning dự đoán chuỗi lợi suất 14 ngày tiếp theo. Sau đó hệ thống đánh giá mô hình và mô phỏng backtest để xem nếu dùng dự đoán đó để giao dịch thì hiệu quả ra sao.

### 1.2 Vì sao em chọn bài toán dự đoán cổ phiếu?

Dạ, cổ phiếu là dữ liệu chuỗi thời gian có tính thực tế cao, nhiều nhiễu và khó dự đoán. Vì vậy bài toán này phù hợp để áp dụng cả Machine Learning và Deep Learning. Ngoài ra, kết quả không chỉ đánh giá bằng sai số dự đoán mà còn có thể kiểm tra bằng backtest giao dịch, giúp đề tài có tính ứng dụng hơn.

### 1.3 Điểm chính của hệ thống StockSight là gì?

Dạ, hệ thống có pipeline khá đầy đủ: lấy dữ liệu từ Yahoo Finance, tạo đặc trưng kỹ thuật và vĩ mô, tạo target, chia train/test theo thời gian, train các mô hình ML và DL, đánh giá bằng nhiều metric, lưu kết quả, backtest và hiển thị bằng Streamlit.

### 1.4 Dự án dự đoán giá trực tiếp hay dự đoán return?

Dạ, dự án chủ yếu dự đoán return, tức tỷ lệ tăng hoặc giảm của giá. Sau khi có return dự đoán, hệ thống mới quy đổi thành giá bằng cách nhân với giá đóng cửa hiện tại. Cách này phù hợp hơn vì giá tuyệt đối của từng cổ phiếu có thang đo khác nhau, còn return dễ so sánh và ổn định hơn.

## 2. Dữ liệu

### 2.1 Dữ liệu được lấy từ đâu?

Dạ, dữ liệu được lấy từ Yahoo Finance thông qua thư viện `yfinance`. Dự án lấy dữ liệu OHLCV của 21 mã cổ phiếu Mỹ và các chỉ số vĩ mô như VIX, lợi suất trái phiếu Mỹ 10 năm, dầu thô và chỉ số USD.

### 2.2 OHLCV là gì?

Dạ, OHLCV gồm `Open`, `High`, `Low`, `Close`, `Volume`. Trong đó `Close` là giá đóng cửa cuối ngày, được dự án dùng làm giá chính để tính return và đánh giá dự báo.

### 2.3 Vì sao cần dùng thêm chỉ số vĩ mô?

Dạ, giá cổ phiếu không chỉ bị ảnh hưởng bởi lịch sử giá của chính nó mà còn bị ảnh hưởng bởi thị trường chung và yếu tố vĩ mô. Ví dụ VIX phản ánh mức độ biến động thị trường, lãi suất ảnh hưởng đến nhóm tài chính và tiện ích, dầu ảnh hưởng đến nhóm năng lượng và công nghiệp.

### 2.4 Dữ liệu được lưu lại như thế nào?

Dạ, sau khi tải bằng `yfinance`, dữ liệu được lưu thành file parquet trong thư mục `data/raw/stocks` và `data/raw/macro`. Việc cache giúp lần chạy sau không cần tải lại dữ liệu qua mạng, làm pipeline nhanh và ổn định hơn.

### 2.5 `auto_adjust=True` trong `yfinance` có ý nghĩa gì?

Dạ, `auto_adjust=True` giúp giá được điều chỉnh theo các sự kiện như chia tách cổ phiếu hoặc cổ tức. Điều này làm chuỗi giá lịch sử nhất quán hơn khi tính return và train mô hình.

## 3. Feature Engineering

### 3.1 Feature engineering là gì?

Dạ, feature engineering là quá trình tạo ra các biến đầu vào có ý nghĩa từ dữ liệu gốc. Với cổ phiếu, từ giá và volume ban đầu, em tạo thêm return, moving average, RSI, MACD, volatility, Bollinger width, volume ratio và các feature vĩ mô.

### 3.2 Các feature kỹ thuật chính trong dự án là gì?

Dạ, các feature kỹ thuật gồm return, log return, phần trăm thay đổi giá so với 1 và 5 ngày trước, MA5, MA20, EMA12, EMA26, RSI14, MACD, MACD signal, volatility 21 ngày, Bollinger width và volume ratio.

### 3.3 RSI là gì?

Dạ, RSI là Relative Strength Index, chỉ báo sức mạnh tương đối. Nó đo mức độ tăng giảm gần đây của giá, thường dùng để nhận biết trạng thái quá mua hoặc quá bán. Trong dự án, RSI là một feature để mô hình tự học quan hệ với return tương lai.

### 3.4 MACD là gì?

Dạ, MACD là hiệu giữa EMA nhanh và EMA chậm, cụ thể trong dự án là EMA12 trừ EMA26. MACD giúp nhận biết động lượng và xu hướng giá. Ngoài MACD, dự án còn tính `macd_signal`, là đường tín hiệu của MACD.

### 3.5 Volatility là gì?

Dạ, volatility là độ biến động của return. Trong dự án, `volatility_21` là độ lệch chuẩn của return trong 21 ngày, tương đương khoảng một tháng giao dịch. Volatility cao nghĩa là giá dao động mạnh hơn.

### 3.6 Bollinger width là gì?

Dạ, Bollinger width đo độ rộng của Bollinger Bands quanh đường trung bình động. Khi dải Bollinger rộng, thị trường đang biến động mạnh hơn. Đây là một feature giúp mô hình nhận biết giai đoạn biến động thấp hoặc cao.

### 3.7 Vì sao feature được chia theo ngành?

Dạ, mỗi ngành có yếu tố ảnh hưởng khác nhau. Ví dụ nhóm năng lượng thường nhạy với giá dầu, còn nhóm tài chính và tiện ích thường nhạy với lãi suất. Vì vậy dự án dùng `sector_config.py` để thêm feature tương quan với dầu hoặc lãi suất tùy theo ngành của ticker.

### 3.8 Có nguy cơ feature dùng dữ liệu tương lai không?

Dạ, dự án hạn chế điều này bằng cách dùng các phép `rolling`, `pct_change`, `shift` theo hướng chỉ sử dụng dữ liệu hiện tại và quá khứ. Ngoài ra, split dữ liệu theo thời gian và scaler chỉ fit trên train để tránh data leakage.

## 4. Target và Preprocessing

### 4.1 Target của Machine Learning là gì?

Dạ, target của Machine Learning là `y_return`, tức return ngày tiếp theo:

```text
y_return = close_ngày_mai / close_hôm_nay - 1
```

Mô hình dùng feature của hôm nay để dự đoán return của ngày mai.

### 4.2 Vì sao dòng cuối thường bị loại bỏ?

Dạ, vì target cần giá ngày mai. Dòng cuối cùng trong dữ liệu không có ngày tiếp theo nên `y_return` bị `NaN`. Sau đó pipeline dùng `dropna()` để loại bỏ dòng này.

### 4.3 Train/test split được thực hiện như thế nào?

Dạ, dữ liệu được sắp xếp theo ngày rồi chia theo thời gian. 85% dữ liệu đầu dùng cho train/validation, 15% cuối dùng cho test. Trong phần train, 10% cuối được dùng làm validation.

### 4.4 Vì sao không dùng random split?

Dạ, vì dữ liệu cổ phiếu là chuỗi thời gian. Nếu random split, mô hình có thể học từ dữ liệu tương lai để dự đoán quá khứ. Điều này không đúng với thực tế và gây data leakage.

### 4.5 MinMaxScaler dùng để làm gì?

Dạ, `MinMaxScaler` đưa các feature về cùng thang đo, thường gần khoảng 0 đến 1. Điều này giúp mô hình học ổn định hơn, đặc biệt là Deep Learning.

### 4.6 Vì sao scaler chỉ fit trên train?

Dạ, nếu fit scaler trên toàn bộ dữ liệu, min và max của test sẽ bị dùng trong train. Đó là một dạng data leakage. Vì vậy dự án chỉ fit scaler trên train, sau đó dùng scaler đó transform validation và test.

### 4.7 Dữ liệu cho Deep Learning được tạo như thế nào?

Dạ, Deep Learning dùng sliding window. Với `window=60` và `horizon=14`, mỗi sample gồm 60 ngày feature quá khứ làm input và 14 ngày return tiếp theo làm target.

## 5. Machine Learning Models

### 5.1 Dự án dùng những mô hình Machine Learning nào?

Dạ, dự án dùng ba mô hình hồi quy: Ridge Regression, Random Forest Regressor và XGBoost Regressor. Ngoài ra còn có hai baseline là `naive_zero` và `naive_persistence`.

### 5.2 Ridge Regression là gì?

Dạ, Ridge Regression là hồi quy tuyến tính có regularization L2. Nó phạt các hệ số quá lớn để giảm overfitting. Trong dự án, Ridge được dùng như một mô hình đơn giản và làm mốc so sánh.

### 5.3 Random Forest Regressor là gì?

Dạ, Random Forest là mô hình ensemble gồm nhiều cây quyết định. Mỗi cây học trên một phần dữ liệu, sau đó kết quả cuối là trung bình dự đoán của các cây. Nó có khả năng học quan hệ phi tuyến tốt hơn mô hình tuyến tính.

### 5.4 XGBoost Regressor là gì?

Dạ, XGBoost là mô hình boosting dựa trên cây quyết định. Các cây được xây tuần tự, cây sau học để sửa lỗi của cây trước. XGBoost thường rất mạnh trên dữ liệu dạng bảng và có thể học quan hệ phi tuyến giữa các feature.

### 5.5 Vì sao cần baseline?

Dạ, baseline giúp kiểm tra mô hình phức tạp có thật sự tốt hơn cách dự đoán đơn giản không. Nếu model không tốt hơn `naive_zero` hoặc `naive_persistence`, thì model chưa chứng minh được giá trị.

### 5.6 `naive_zero` là gì?

Dạ, `naive_zero` là baseline luôn dự đoán return ngày mai bằng 0. Nghĩa là giả sử giá ngày mai không đổi.

### 5.7 `naive_persistence` là gì?

Dạ, `naive_persistence` dự đoán return ngày mai bằng return hôm nay. Đây là cách giả định xu hướng rất ngắn hạn tiếp tục lặp lại.

### 5.8 ML dự đoán nhiều ngày như thế nào?

Dạ, vì ML chỉ train để dự đoán 1 ngày, nên khi muốn dự đoán 14 ngày, dự án dùng recursive forecast. Tức là dự đoán ngày 1, cập nhật các feature lag, rồi dùng kết quả đó dự đoán ngày 2, lặp lại đến ngày 14.

## 6. Deep Learning Models

### 6.1 Mô hình Deep Learning trong dự án là gì?

Dạ, mô hình Deep Learning là CNN + BiLSTM Encoder + LSTM Decoder + Attention. Input là chuỗi 60 ngày feature, output là chuỗi 14 ngày return.

### 6.2 Vì sao dùng CNN trong dữ liệu chuỗi thời gian?

Dạ, Conv1D có thể phát hiện các pattern cục bộ trong chuỗi, ví dụ biến động ngắn hạn, thay đổi momentum hoặc volatility. Nó hoạt động như một bộ trích xuất đặc trưng trước khi đưa vào LSTM.

### 6.3 LSTM là gì?

Dạ, LSTM là một dạng Recurrent Neural Network được thiết kế để học dữ liệu chuỗi. Nó có các cổng giúp quyết định thông tin nào cần ghi nhớ, cập nhật hoặc quên, nên phù hợp với bài toán chuỗi thời gian.

### 6.4 BiLSTM là gì?

Dạ, BiLSTM là LSTM hai chiều. Nó đọc chuỗi trong cửa sổ input theo cả hai hướng để học biểu diễn đầy đủ hơn. Trong dự án, BiLSTM chỉ nhìn trong 60 ngày quá khứ đã đưa vào input, không dùng dữ liệu tương lai của test.

### 6.5 Attention có vai trò gì?

Dạ, Attention giúp mô hình tập trung vào những ngày quan trọng hơn trong chuỗi 60 ngày. Không phải ngày nào trong quá khứ cũng có cùng mức ảnh hưởng đến dự đoán, nên attention giúp mô hình gán trọng số cao hơn cho các thời điểm có tín hiệu mạnh.

### 6.6 Seq2Seq là gì?

Dạ, Seq2Seq là kiến trúc biến một chuỗi đầu vào thành một chuỗi đầu ra. Trong dự án, chuỗi đầu vào là 60 ngày feature, chuỗi đầu ra là 14 ngày return tương lai.

### 6.7 Directional loss là gì?

Dạ, directional loss là MSE nhưng nếu mô hình dự đoán sai hướng tăng/giảm thì lỗi bị nhân thêm penalty. Cách này giúp mô hình quan tâm nhiều hơn đến việc dự đoán đúng dấu âm/dương của return.

### 6.8 Vì sao output DL bị giới hạn bằng `tanh`?

Dạ, output dùng `tanh * return_cap`, với `return_cap` mặc định là 0.05. Như vậy return dự đoán mỗi ngày được giới hạn quanh -5% đến +5%, tránh mô hình đưa ra dự đoán quá cực đoan.

## 7. Đánh giá mô hình

### 7.1 MAE là gì?

Dạ, MAE là sai số tuyệt đối trung bình:

```text
MAE = trung bình |y_true - y_pred|
```

MAE càng nhỏ thì mô hình càng dự đoán gần giá trị thật.

### 7.2 RMSE là gì?

Dạ, RMSE là căn bậc hai của sai số bình phương trung bình. RMSE phạt nặng các lỗi lớn hơn MAE, nên phù hợp để phát hiện model có nhiều dự đoán sai mạnh.

### 7.3 Directional accuracy là gì?

Dạ, directional accuracy là tỷ lệ mô hình dự đoán đúng hướng tăng hoặc giảm:

```text
sign(y_pred) == sign(y_true)
```

Trong chứng khoán, metric này quan trọng vì quyết định mua/bán thường phụ thuộc vào hướng tăng giảm.

### 7.4 MAPE là gì?

Dạ, MAPE là sai số phần trăm tuyệt đối trung bình. Với DL, sau khi quy đổi return thành giá, MAPE cho biết dự đoán giá lệch trung bình bao nhiêu phần trăm so với giá thật.

### 7.5 R2 score có ý nghĩa gì?

Dạ, R2 đo mức độ mô hình giải thích được phương sai của target. Tuy nhiên với dữ liệu tài chính, R2 thường thấp vì return có nhiều nhiễu và rất khó dự đoán.

### 7.6 Vì sao cần đánh giá cả validation và test?

Dạ, validation giúp theo dõi mô hình trong giai đoạn train hoặc so sánh trong tập gần train, còn test là dữ liệu mới hơn dùng để đánh giá cuối cùng. Nếu validation tốt nhưng test kém, mô hình có thể bị overfit hoặc thị trường đã thay đổi.

## 8. Backtest

### 8.1 Backtest là gì?

Dạ, backtest là mô phỏng chiến lược giao dịch trên dữ liệu quá khứ. Nó giúp kiểm tra nếu dùng tín hiệu dự đoán của mô hình để đầu tư thì lợi nhuận và rủi ro sẽ như thế nào.

### 8.2 Chiến lược top-k trong dự án là gì?

Dạ, mỗi ngày hệ thống chọn tối đa `top_k` cổ phiếu có tín hiệu dự đoán cao hơn threshold. Các cổ phiếu được chọn có trọng số bằng nhau, phần vốn còn lại giữ tiền mặt nếu không đủ cổ phiếu đạt ngưỡng.

### 8.3 Vì sao backtest phải shift tín hiệu 1 ngày?

Dạ, để tránh look-ahead bias. Tín hiệu ở ngày `t` chỉ được dùng để nắm giữ vị thế từ ngày `t+1`, vì trong thực tế ta không thể dùng kết quả cuối ngày để giao dịch ngay trước đó.

### 8.4 Chi phí giao dịch được tính như thế nào?

Dạ, chi phí giao dịch được tính theo mức thay đổi trọng số danh mục:

```text
cost = cost_per_trade * sum(|w_t - w_(t-1)|)
```

Nếu danh mục thay đổi vị thế nhiều thì chi phí cao hơn.

### 8.5 Sharpe ratio là gì?

Dạ, Sharpe ratio là chỉ số đo lợi nhuận trên rủi ro. Trong dự án, Sharpe được tính bằng lợi nhuận năm hóa chia cho volatility năm hóa. Sharpe càng cao thì chiến lược càng tốt về mặt lợi nhuận điều chỉnh theo rủi ro.

### 8.6 Max drawdown là gì?

Dạ, max drawdown là mức sụt giảm lớn nhất từ đỉnh xuống đáy của đường equity. Đây là metric rủi ro rất quan trọng vì cho biết chiến lược có thể lỗ sâu nhất bao nhiêu trong giai đoạn backtest.

## 9. Streamlit và trực quan hóa

### 9.1 Giao diện Streamlit gồm những trang nào?

Dạ, giao diện gồm Home, Predict, Compare & Backtest và Explain. Home hiển thị tổng quan ticker, Predict chạy dự báo live, Compare & Backtest so sánh mô hình và mô phỏng giao dịch, Explain dùng SHAP để giải thích mô hình.

### 9.2 Trang Predict hoạt động như thế nào?

Dạ, người dùng chọn ticker, model và horizon. Hệ thống load model đã train, tạo feature mới nhất, dự đoán return, quy đổi thành đường giá dự báo, sau đó đưa ra khuyến nghị BUY, HOLD hoặc SELL.

### 9.3 Khuyến nghị BUY/HOLD/SELL được tạo như thế nào?

Dạ, hệ thống tính cumulative return trong horizon. Nếu cumulative return lớn hơn threshold thì BUY, nếu nhỏ hơn âm threshold thì SELL, còn nằm giữa thì HOLD.

### 9.4 Trang Explain dùng gì để giải thích mô hình?

Dạ, trang Explain dùng SHAP cho Random Forest và XGBoost. SHAP cho biết feature nào đóng góp làm dự đoán tăng hoặc giảm.

## 10. Hạn chế và hướng phát triển

### 10.1 Hạn chế lớn nhất của đề tài là gì?

Dạ, hạn chế lớn nhất là dữ liệu chủ yếu là dữ liệu giá và một số chỉ số vĩ mô, chưa có tin tức, sentiment, báo cáo tài chính hoặc dữ liệu cơ bản doanh nghiệp. Ngoài ra thị trường tài chính nhiều nhiễu nên dự đoán chính xác là rất khó.

### 10.2 ML recursive forecast có hạn chế gì?

Dạ, ML recursive forecast chỉ cập nhật một số feature lag như return, log return, close_pct_lag_1 và close_pct_lag_5. Các feature phức tạp như RSI, MACD, moving average chưa được tính lại đầy đủ cho tương lai.

### 10.3 Backtest có hạn chế gì?

Dạ, backtest dùng giả định đơn giản về giao dịch, như chi phí cố định và chưa mô phỏng đầy đủ slippage hoặc thanh khoản. Vì vậy kết quả backtest chỉ nên xem như mô phỏng nghiên cứu, chưa phải khuyến nghị đầu tư thực tế.

### 10.4 Hướng phát triển tiếp theo là gì?

Dạ, có thể mở rộng bằng cách thêm dữ liệu tin tức, sentiment, dữ liệu tài chính doanh nghiệp, dùng walk-forward validation, tuning hyperparameter kỹ hơn, thêm quản trị rủi ro và thử các mô hình sequence hiện đại hơn như Transformer.

## 11. Câu hỏi phản biện khó

### 11.1 Nếu mô hình dự đoán tốt theo MAE nhưng backtest kém thì sao?

Dạ, điều đó có thể xảy ra vì metric dự đoán và hiệu quả giao dịch không hoàn toàn giống nhau. MAE đo sai số trung bình, còn backtest phụ thuộc vào hướng dự đoán, thời điểm vào lệnh, chi phí giao dịch và quản trị rủi ro. Vì vậy dự án đánh giá cả metric ML và backtest.

### 11.2 Vì sao directional accuracy chỉ hơn 50% một chút vẫn có ý nghĩa?

Dạ, trong thị trường tài chính, dữ liệu rất nhiễu nên chỉ cần lợi thế nhỏ nhưng ổn định cũng có thể có ý nghĩa nếu kết hợp với quản trị rủi ro và chi phí hợp lý. Tuy nhiên, nếu directional accuracy chỉ nhỉnh hơn 50%, vẫn cần kiểm tra bằng backtest để xem có thực sự sinh lợi không.

### 11.3 Nếu test set có giai đoạn thị trường khác train set thì sao?

Dạ, đó là vấn đề regime shift. Mô hình học từ dữ liệu quá khứ có thể kém hiệu quả khi thị trường thay đổi mạnh. Đây là lý do cần đánh giá trên test theo thời gian và có thể dùng walk-forward retraining trong hướng phát triển.

### 11.4 Vì sao không dùng trực tiếp giá đóng cửa làm feature?

Dạ, trong danh sách feature đầu vào, project loại `close` khỏi input chính vì giá tuyệt đối có thang đo khác nhau giữa các ticker. Model học return và các chỉ báo tương đối. Tuy nhiên `close` vẫn được giữ lại để quy đổi return dự đoán thành đường giá.

### 11.5 Vì sao Deep Learning không chắc chắn tốt hơn Machine Learning?

Dạ, Deep Learning mạnh hơn trong học chuỗi nhưng cũng cần nhiều dữ liệu, train lâu và dễ overfit. Với dữ liệu tài chính nhiều nhiễu, mô hình phức tạp chưa chắc luôn tốt hơn. Vì vậy dự án so sánh DL với ML và baseline.

