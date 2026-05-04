# Quy trình Xây dựng Tensor 3D và Phân tích Kết nối (Cập nhật 34 Subjects)
## 🧠 Dựa trên phương pháp Ozdemir (2017)

Sau khi tiền xử lý, dữ liệu EEG sạch được chuyển đổi thành các Tensor 3D để đại diện cho sự thay đổi của mạng lưới kết nối não bộ theo thời gian.

---

## 1. Cơ sở lý thuyết: Phân tích Kết nối (Connectivity)
Mạng lưới não bộ không chỉ được xác định bởi biên độ tín hiệu tại một điểm, mà bởi sự **đồng bộ pha** giữa các vùng não khác nhau. 

### Kỹ thuật Ước lượng Pha: RID-Rihaczek
Dự án sử dụng **Reduced Interference Rihaczek Distribution (RID-Rihaczek)** để ước lượng pha tức thời của tín hiệu EEG trong miền thời gian-tần số.
- **Ưu điểm**: Độ phân giải thời gian-tần số cực cao, giảm thiểu nhiễu chéo (cross-terms) so với các phương pháp truyền thống như Wigner-Ville hay Wavelet.
- **Dải tần**: Tập trung vào dải **Theta (4-8 Hz)** – dải tần số đặc trưng cho các quá trình kiểm soát nhận thức và phát hiện lỗi trong não bộ.

---

## 2. Chỉ số Kết nối: PLV (Phase Locking Value)
Để đo lường sự kết nối giữa hai kênh EEG (nút mạng), dự án sử dụng **Phase Locking Value (PLV)**:
- **Ý nghĩa**: Đo lường mức độ ổn định của sự lệch pha giữa hai tín hiệu trên nhiều thử nghiệm (trials).
- **Giá trị**: Từ 0 (hoàn toàn không đồng bộ) đến 1 (đồng bộ hoàn hảo).
- **Công thức**: $PLV_{i,j}(t) = \frac{1}{K} |\sum_{k=1}^{K} e^{j(\phi_i^k(t) - \phi_j^k(t))}|$ (với $K$ là số lượng trial).

---

## 3. Cấu trúc Tensor 3D/4D sau cùng
Dữ liệu từ **34 đối tượng** đã được tổng hợp thành một khối Tensor lớn:

### Tensor tổng hợp (4D):
- **Kích thước**: `(34, 30, 30, 256)`
    - `34`: Số lượng người tham gia đạt tiêu chuẩn (>15 lỗi).
    - `30 x 30`: Ma trận kết nối đầy đủ giữa tất cả các cặp kênh EEG.
    - `256`: Các mốc thời gian (từ -1.0s đến 1.0s, tương ứng 128Hz).

---

## 4. Quy trình thực hiện (Pipeline)
1. **Cân bằng Trial (Trial Balancing)**: 
    - Vì số lượng Correct >> Incorrect, chúng ta thực hiện lấy mẫu ngẫu nhiên (Random Subsampling) các trial Correct để có tỉ lệ 1:1. 
    - Điều này đảm bảo giá trị PLV không bị thiên kiến do số lượng mẫu khác nhau.
2. **GPU Acceleration**: 
    - Sử dụng **PyTorch** để tính toán RID-Rihaczek trên GPU. 
    - Việc tính toán 30x30 ma trận PLV cho 256 thời điểm x 34 người là cực kỳ nặng nề, GPU giúp giảm thời gian từ vài giờ xuống còn vài phút.
3. **Lưu trữ**: 
    - `tensor_correct_4d.npy` (~30MB)
    - `tensor_incorrect_4d.npy` (~30MB)

---

## 5. Ý nghĩa của Tensor trong bài báo Ozdemir
Việc xây dựng Tensor này cho phép chúng ta áp dụng các thuật toán **Recursive Tensor Subspace Tracking (HO-RLSL)**. Thay vì nhìn vào từng ma trận kết nối riêng lẻ, thuật toán sẽ nhìn vào toàn bộ khối dữ liệu để tìm ra:
- **Cấu trúc cộng đồng ổn định**: Các vùng não thường xuyên "nói chuyện" với nhau.
- **Điểm thay đổi động (Change-points)**: Thời điểm chính xác mà mạng lưới não bộ tái cấu trúc để phản ứng với lỗi lầm.

---
*Tài liệu được cập nhật dựa trên kết quả chạy 40 subjects (04/05/2026).*
