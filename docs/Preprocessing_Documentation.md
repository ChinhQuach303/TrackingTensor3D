# Tài liệu Chi tiết Quy trình Tiền xử lý (Preprocessing Pipeline)
## 🧠 Dự án: TrackingTensor3D | Dữ liệu: 40 Subjects (ERP CORE)

Tài liệu này cung cấp cái nhìn chuyên sâu vào các kỹ thuật xử lý tín hiệu số được áp dụng để chuyển đổi dữ liệu EEG thô từ 40 người tham gia thành các đoạn tín hiệu sạch (Clean Epochs) phục vụ cho phân tích Tensor.

---

## 1. Kiến trúc Quy trình Tiền xử lý
Quy trình được thực hiện thông qua tập lệnh `src/master_preprocessing.py` với các tham số được tối ưu hóa cho thành phần ERN:

### A. Chuẩn hóa và Lọc tín hiệu (Standardization & Filtering)
- **Resampling (128 Hz)**: Dữ liệu gốc từ ERP CORE thường có tốc độ lấy mẫu cao. Việc giảm xuống 128 Hz là bước tối ưu để:
    - Giảm nhiễu cao tần không cần thiết.
    - Tiết kiệm bộ nhớ khi tính toán Tensor 4D (vốn rất nặng).
- **Lọc thông dải (Band-pass Filter)**: Sử dụng bộ lọc **Zero-phase FIR (firwin)**.
    - **0.1 Hz (High-pass)**: Loại bỏ các thành phần điện một chiều (DC offset) và nhiễu trôi chậm do mồ hôi hoặc chuyển động nhẹ.
    - **30 Hz (Low-pass)**: Loại bỏ nhiễu điện lưới (50/60 Hz) và nhiễu cơ (EMG). Dải tần này bao phủ hoàn toàn dải **Theta (4-8 Hz)** trọng tâm của nghiên cứu.

### B. Loại bỏ nhiễu bằng ICA (Independent Component Analysis)
Hệ thống sử dụng thuật toán **Extended Infomax ICA** để tách tín hiệu thành 10 thành phần độc lập (ICs):
- **Cơ chế tự động**: Thay vì quan sát thủ công, hệ thống sử dụng phương pháp thống kê để tìm các IC có tương quan cao với kênh mắt `Fp1`.
- **Rejection**: Các thành phần được xác định là nhiễu mắt (blinks) sẽ bị loại bỏ hoàn toàn. Điều này giúp làm sạch vùng trán mà không làm thay đổi pha của tín hiệu thần kinh thực sự – một yếu tố sống còn cho phân tích PLV sau này.

### C. Biến đổi mật độ nguồn dòng (Current Source Density - CSD)
Đây là bước đột phá kỹ thuật trong quy trình của Ozdemir:
- **Nguyên lý**: Sử dụng thuật toán Spherical Spline để tính toán Laplacian bề mặt của điện thế da đầu.
- **Tại sao cần CSD?**: 
    - Tín hiệu EEG thô bị "mờ" do sự dẫn truyền qua hộp sọ (Volume Conduction). 
    - CSD hoạt động như một bộ lọc không gian, làm "sắc nét" các hoạt động điện ngay bên dưới điện cực. 
    - Nó biến các kênh EEG thành các nguồn phát độc lập, giúp ma trận kết nối mạng lưới (Connectivity) phản ánh chính xác tương tác giữa các vùng não thay vì chỉ là sự rò rỉ tín hiệu.

---

## 2. Phân đoạn và Gán nhãn (Epoching & Labeling)
Dữ liệu sau khi sạch được cắt thành các đoạn nhỏ (Epochs) dựa trên các mốc thời gian phản ứng:
- **Thời gian**: từ **-1.0s đến +1.0s** (tổng cộng 2s, 256 điểm dữ liệu).
- **Phân loại**:
    - **Correct (Làm đúng)**: Các thử nghiệm có mã sự kiện phản hồi khớp với kích thích (ví dụ: kích thích trái - phản hồi trái).
    - **Incorrect (Làm sai)**: Các thử nghiệm có sự sai lệch giữa phản hồi và kích thích (đây là nơi phát sinh tín hiệu ERN).
- **Detrending & Baseline**: Tín hiệu được loại bỏ xu hướng tuyến tính (detrend) và hiệu chỉnh nền dựa trên khoảng **-600ms đến -400ms** trước khi phản ứng.

---

## 3. Thống kê Quy mô 40 Subjects
- **Tỉ lệ thành công**: 100% (40/40 người được xử lý mà không có lỗi hệ thống).
- **Tính nhất quán**: Tất cả 40 người đều sử dụng cùng một Montage (`standard_1005`) và 30 kênh EEG tiêu chuẩn, đảm bảo tính đồng nhất khi gộp vào Tensor 4D.
- **Sản phẩm đầu ra**: Mỗi người tạo ra một tệp `*_master-epo.fif` chứa đầy đủ các thông tin đã xử lý.

---

## 4. Tóm tắt các thông số kỹ thuật (Config)

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **SFREQ** | 128 Hz | Tốc độ lấy mẫu mục tiêu |
| **Filter Range** | 0.1 - 30.0 Hz | FIR Zero-phase |
| **ICA Method** | Extended Infomax | 10 components |
| **Reference** | Current Source Density (CSD) | Laplacian transform |
| **Epoch Window** | [-1.0, 1.0] s | 256 time points |
| **Baseline** | [-0.6, -0.4] s | Trước khi phản ứng |

---
> [!IMPORTANT]
> Toàn bộ quy trình này được thiết kế để bảo toàn **thông tin pha (phase information)** của tín hiệu. Đây là yếu tố tiên quyết vì bước tiếp theo (Connectivity) sẽ sử dụng sự lệch pha để đo lường mức độ kết nối của mạng lưới não bộ.

---
*Tài liệu được cập nhật chi tiết dựa trên dữ liệu thực tế 40 subjects (04/05/2026).*
