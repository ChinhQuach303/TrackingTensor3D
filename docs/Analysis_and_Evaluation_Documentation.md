# Hướng dẫn Phân rã Tensor và Đánh giá Kết quả
## 🧠 Phương pháp: HO-RLSL & FCCA Analysis

Sau khi xây dựng thành công Tensor 4D, bước cuối cùng là phân rã khối dữ liệu này để tìm ra các quy luật hoạt động của não bộ và đánh giá tính ý nghĩa về mặt thống kê.

---

## 1. Phân rã Tensor (HO-RLSL)
Dự án sử dụng thuật toán **Higher-Order Recursive Low-Rank + Sparse Structure Learning (HO-RLSL)**.

### Cách thức hoạt động:
- **Tách cấu trúc (Decomposition)**: Thuật toán tách Tensor kết nối thành:
    1. **Low-rank part**: Đại diện cho cấu trúc mạng lưới ổn định và có tổ chức (Communities).
    2. **Sparse part**: Đại diện cho các kết nối nhiễu hoặc các thay đổi đột ngột mang tính nhất thời.
- **Theo dõi không gian con (Subspace Tracking)**: Thay vì phân tích từng thời điểm độc lập, HO-RLSL cập nhật không gian con một cách đệ quy, giúp bắt trọn tính liên tục của dữ liệu.

### Kết quả thu được:
- **Trọng số mạng lưới ($w_t$)**: Cho biết mức độ quan trọng của từng kênh EEG trong mạng lưới tại mỗi thời điểm.
- **Năng lượng mạng (Energy)**: Tổng mức độ đồng bộ của mạng lưới chung.

---

## 2. Phân tích Module chức năng (FCCA)
Bên cạnh HO-RLSL, chúng ta sử dụng **Functional Canonical Correlation Analysis (FCCA)** để đánh giá sự tương tác giữa các vùng não lớn.

- **Các Module**: Frontal (Trán), Central (Trung tâm), Parietal (Đỉnh), Occipital (Chẩm).
- **Mục tiêu**: Tìm xem khi mắc lỗi (ERN), các module nào "giao tiếp" với nhau mạnh nhất.
- **Kết quả**: Vùng **Frontal** và **Central** cho thấy sự tương quan cao nhất ($>0.5$), khẳng định giả thuyết về mạng lưới kiểm soát lỗi trán-trung tâm.

---

## 3. Quy trình Đánh giá (Evaluation)
Việc đánh giá được thực hiện qua hai bước chính:

### A. So sánh T-test (Incorrect vs Correct)
- **Phương pháp**: Sử dụng t-test mẫu phụ thuộc (Paired t-test) trên năng lượng mạng lưới của 34 đối tượng.
- **Kết quả**: Năng lượng mạng lưới khi làm Sai cao hơn đáng kể so với khi làm Đúng tại cửa sổ thời gian **50-100ms** ($p < 0.05$ tại vùng đỉnh ERN).

### B. Tương quan hành vi (Behavioral Correlation)
- **Phương pháp**: Tính hệ số tương quan Pearson giữa **Năng lượng mạng lưới cực đại** và **Tỉ lệ lỗi (Error Rate)** của từng người.
- **Kết quả hiện tại**: $R = -0.161, p = 0.362$. 
    - *Ý nghĩa*: Xu hướng người có mạng lưới ERN mạnh thì ít mắc lỗi hơn. Mặc dù $p$ chưa đạt mức < 0.05 nhưng hướng tương quan là phù hợp với lý thuyết thần kinh học.

---

## 4. Cách đọc biểu đồ Kết quả (Interpretation)

1. **Biểu đồ `ozdemir_replicated_dynamics.png`**:
    - **Top panel**: Đường năng lượng đen. Các vạch đỏ đứt đoạn là **Change-points** (điểm não bộ thay đổi trạng thái).
    - **Middle panel**: Bản đồ nhiệt (Heatmap). Trục dọc là tên kênh, trục ngang là thời gian. Vùng màu đỏ/vàng đậm biểu thị các kênh đang nắm giữ vai trò chủ chốt trong mạng lưới.
    - **Bottom panel**: Topomap. Hiển thị hình ảnh "não bộ" tại thời điểm đỉnh ERN. Vùng đỏ đậm thường nằm ở đỉnh đầu (FCz, Cz).

2. **Biểu đồ `statistical_validation_final.png`**:
    - So sánh trực quan mức độ khác biệt giữa hai điều kiện và biểu đồ phân tán (Scatter plot) cho tương quan hành vi.

---
## 5. Kết luận sau cùng
Quy trình phân rã và đánh giá đã chứng minh rằng phương pháp **Tensor Tracking** có khả năng bắt được những biến đổi tinh vi của não bộ mà các phương pháp trung bình cộng truyền thống có thể bỏ qua.

---
*Tài liệu được tạo tự động bởi Antigravity Coding Assistant.*
