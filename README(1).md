# Phân tích mạng lưới não bộ động (dFCN) cho dữ liệu ERN bằng Tensor Subspace Tracking

Dự án này tái hiện quy trình từ bài báo: *"Recursive Tensor Subspace Tracking for Dynamic Brain Network Analysis"* (Ozdemir et al., 2017) áp dụng trên bộ dữ liệu **ERP CORE ERN**.

## 🚀 Chiến lược tối ưu tài nguyên
Để xử lý 40 đối tượng với CPU/RAM tiêu chuẩn, kiến trúc áp dụng:
*   **Downsampling:** Hạ tần số lấy mẫu từ 1024Hz xuống **128Hz**. và 2s.
*   **Subject-wise Processing:** Tiền xử lý sạch dữ liệu của từng người trước khi gộp.
*   **Memory Mapping:** Sử dụng `numpy.memmap` cho các Tensor lớn để tránh lỗi Out-of-Memory.
*   **Precision:** Sử dụng `float32` thay vì `float64`.

## 🛠 Project Pipeline

### Giai đoạn 1: Tiền xử lý cá nhân (Subject-wise Preprocessing)
Thực hiện tuần tự trên từng `sub-XXX`:
1.  **Lọc dải tần (Band-pass Filter):** Tập trung vào dải **Theta (4-8 Hz)** theo yêu cầu của nghiên cứu kết nối ERN.
2.  **Hạ tần số (Resampling):** Chuyển về **128 Hz**.
3.  **Cắt đoạn (Epoching):** Trích xuất tín hiệu quanh thời điểm phản hồi (Response-locked) từ **-600ms đến 400ms**.
4.  **Lọc nhiễu:** Loại bỏ các trial bị lỗi dựa trên thông tin từ file `events.tsv`.
5.  **Lưu trữ:** Lưu Epochs đã làm sạch của từng người vào file trung gian để giải phóng RAM.

### Giai đoạn 2: Tính toán ma trận kết nối (Connectivity Estimation)
1.  Với mỗi Epoch, tính toán ma trận thông tin **Phase Locking Value (PLV)** giữa 33 điện cực.
2.  Kết quả đầu ra là một chuỗi ma trận kề $(N \times N)$ theo thời gian cho mỗi đối tượng.

### Giai đoạn 3: Xây dựng Tensor 4D
1.  Gộp Adjacency Matrices của tất cả 40 người thành một Tensor 4D khổng lồ: $(N \times N \times T \times S)$.
    *   $N$: 33 điện cực.
    *   $T$: 128 điểm thời gian (cho 1 giây tín hiệu).
    *   $S$: 40 đối tượng nghiên cứu.
2.  Tensor này sẽ được ánh xạ qua đĩa cứng (Memory-mapped) để truy xuất từng "lát cắt" thời gian.

### Giai đoạn 4: Thuật toán HO-RLSL (Hạt nhân của dự án)
Triển khai thuật toán đệ quy của Ozdemir để theo dõi không gian con:
1.  **Initialization:** Dùng dữ liệu baseline (-600ms đến -200ms) để khởi tạo không gian con hạng thấp (Low-rank subspace).
2.  **Recursive Update:** Tại mỗi bước thời gian $t$, cập nhật ước lượng không gian con và tách phần **Low-rank** (Cộng đồng não bộ chung) khỏi phần **Sparse** (Nhiễu).
3.  **Change Point Detection:** Tự động phát hiện các thời điểm mà mạng lưới não bộ chuyển vùng hoạt động.

### Giai đoạn 5: Phân cụm & Giải thích (Clustering & Analysis)
1.  Sử dụng **Fiedler Consensus Clustering (FCCA)** trên các đoạn thời gian chính (Pre-ERN, ERN, Post-ERN).
2.  Vẽ đồ thị mạng lưới não bộ để so sánh sự khác biệt giữa Correct và Incorrect trials.

## 📁 Cấu trúc thư mục dự kiến
- `data/`: Chứa dữ liệu BIDS gốc.
- `src/preprocessing.py`: Script chạy làm sạch dữ liệu.
- `src/connectivity.py`: Tính ma trận PLV.
- `src/ho_rlsl.py`: Triển khai thuật toán Tensor cốt lõi.
- `outputs/`: Chứa các Tensor đã tính toán và hình ảnh kết quả.
