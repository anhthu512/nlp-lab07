# LAB 07 — From Attention to Transformer

Cài đặt Transformer Encoder bằng PyTorch và ứng dụng phân loại cảm xúc văn bản trên IMDb.

## 1. Cấu trúc thư mục

```text
lab07/
├── README.md
├── positional_encoding.py
├── transformer_block.py
├── transformer_classification.ipynb
├── results.csv
├── positional_encoding.png
├── attention_heads.png
├── learning_curves.png
├── confusion_matrix.png
├── data/
└── outputs/
    ├── checkpoints/
    ├── configuration.json
    ├── vocabulary.json
    ├── split_manifest.csv
    ├── runtime_shapes.csv
    ├── learning_history.csv
    ├── application_predictions.csv
    └── ...
```

## 2. Môi trường

- Python 3.10 trở lên.
- PyTorch.
- NumPy, pandas, matplotlib.
- scikit-learn, pyarrow, IPython.
- JupyterLab hoặc VS Code có hỗ trợ Jupyter Notebook.

Cài đặt thư viện:

```bash
python -m pip install torch numpy pandas matplotlib scikit-learn pyarrow ipython jupyterlab ipykernel
```

Notebook tự chọn GPU CUDA nếu có; nếu không sẽ sử dụng CPU.

## 3. Chạy notebook

1. Mở `transformer_classification.ipynb`.
2. Chọn kernel Python đã cài các thư viện.
3. Đặt thư mục làm việc tại `NLP/` hoặc `NLP/lab07/`.
4. Thêm hoặc chạy cell setup sau trước cell cấu hình:

```python
import sys
```

5. Chọn chế độ chạy trong cell cấu hình:

```python
MODE = "full"
```

6. Chọn **Restart Kernel and Run All Cells**.
7. Sau khi chạy xong, lưu notebook để giữ output, bảng kết quả và biểu đồ trong file `.ipynb`.

### Chạy kiểm tra nhanh

Thay chế độ thành:

```python
MODE = "smoke"
```

Chế độ này sử dụng:

- 64 mẫu train.
- 32 mẫu validation.
- 32 mẫu test.
- 1 epoch.
- Batch size 16.

Kết quả kiểm tra nhanh được lưu tại `outputs/smoke/`.

## 4. Cấu hình chạy đầy đủ

| Tham số | Giá trị |
|---|---:|
| Seed | 42 |
| Embedding dimension / d_model | 128 |
| Số Transformer block | 1 |
| Số attention head mặc định | 4 |
| FFN dimension | 512 |
| Dropout | 0.1 |
| Độ dài chuỗi mặc định | 128 |
| Batch size | 64 |
| Epochs | 6 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |

Checkpoint được chọn theo validation loss thấp nhất.

## 5. Dữ liệu

Dataset: IMDb, gồm hai nhãn:

- `0`: Negative.
- `1`: Positive.

Notebook ưu tiên sử dụng lại dữ liệu, split manifest và vocabulary của LAB 05 nếu có.

Nếu không có dữ liệu cục bộ, notebook tải các file IMDb đã cố định phiên bản và kiểm tra SHA-256.

Nếu không có split manifest, notebook chia tập train chính thức thành train/validation theo tỷ lệ 80/20, phân tầng theo nhãn và sử dụng seed 42. Tập test lấy từ split test chính thức sau bước làm sạch.

Vocabulary được xây dựng từ tập train hoặc sử dụng lại vocabulary LAB 05. Token `<CLS>` được bổ sung để chạy thí nghiệm pooling.

## 6. Các thành phần code

### positional_encoding.py

- Hàm `positional_encoding`.
- Class `PositionalEncoding`.
- Sử dụng sin/cos theo công thức của đề.
- Hỗ trợ tensor đầu vào dạng `(batch, tokens, d_model)`.

### transformer_block.py

- Class `MultiHeadSelfAttention`.
- Class `TransformerBlock`.
- Tự ghép các phép projection Q/K/V, attention, concatenation và output projection.
- Thứ tự block:

```text
Multi-Head Self-Attention
→ Residual + LayerNorm
→ Feed-Forward Network
→ Residual + LayerNorm
```

- Hỗ trợ padding mask và ghi lại shape tensor khi chạy.
- Không sử dụng `TransformerEncoderLayer` trong core implementation.

### transformer_classification.ipynb

Pipeline:

```text
Raw text
→ Tokenizer
→ Token IDs
→ Embedding
→ Positional Encoding
→ Transformer Encoder
→ Pooling
→ Linear classifier
→ Prediction
```

## 7. Các thí nghiệm

| Experiment | Nội dung |
|---|---|
| 1 | Positional encoding và các cặp câu hoán vị token |
| 2 | So sánh 1, 2, 4, 8 attention heads với d_model = 128 |
| 3 | Trực quan hóa attention matrix của từng head |
| 4 | So sánh LSTM và Transformer |
| 5 | So sánh max_length = 32, 64, 128, 256 |
| 6 | So sánh mean pooling và CLS pooling |

Các thí nghiệm dùng chung cấu hình sẽ sử dụng lại kết quả trong cùng phiên chạy. Một lần Run All đầy đủ thực hiện 9 cấu hình huấn luyện riêng biệt.

## 8. Kết quả được lưu

- `results.csv`: kết quả tổng hợp các thí nghiệm.
- `positional_encoding.png`: biểu đồ positional encoding.
- `attention_heads.png`: attention matrix của từng head.
- `learning_curves.png`: training loss và validation loss.
- `confusion_matrix.png`: confusion matrix của LSTM và Transformer.
- `outputs/checkpoints/`: trọng số checkpoint được chọn.
- `outputs/history_*.csv`: lịch sử huấn luyện từng cấu hình.
- `outputs/predictions_*.csv`: dự đoán trên tập test.
- `outputs/application_predictions.csv`: dự đoán cho các câu kiểm tra ứng dụng.
- `outputs/runtime_shapes.csv`: shape tensor ghi nhận khi chạy.

Các chỉ số phân loại gồm Accuracy, Precision, Recall và F1. Precision, Recall và F1 sử dụng cách tính binary với lớp Positive là nhãn 1.

Bộ câu kiểm tra ứng dụng gồm câu ngắn, câu dài, phủ định, tương phản và cảm xúc mơ hồ.

## 9. Kiểm tra thực thi

Code có các kiểm tra:

- Shape của Q, K, V và attention scores.
- Shape đầu ra của Transformer block.
- Forward và backward.
- Padding mask và trường hợp toàn padding.
- Giá trị loss hữu hạn.
- Xác suất dự đoán hợp lệ.
- Sự tồn tại của checkpoint, CSV và biểu đồ.

## 10. AI assistance statement

- **Tool:** ChatGPT.
- **Purpose:** Hỗ trợ cài đặt, debugging và kiểm tra thực thi.
- **Generated:** Positional encoding, Transformer block và code notebook.
- **Modified:** Cấu hình thực nghiệm, xử lý padding, lưu kết quả và trực quan hóa.
- **Verified:** Kiểm tra số học và gradient của attention, chạy smoke test và chạy đầy đủ 6 thí nghiệm.
