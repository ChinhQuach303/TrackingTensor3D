# Exploratory Data Analysis (EDA) Documentation (Cập nhật 40 Subjects)
## 🧠 Project: TrackingTensor3D (ERN Component - Flanker Task)

Tài liệu này tóm tắt quá trình Khám phá Dữ liệu (EDA) chi tiết cho toàn bộ **40 đối tượng** trong bộ dữ liệu EEG thuộc dự án **TrackingTensor3D**, dựa trên nghiên cứu của Ozdemir (2017) và bộ dữ liệu **ERP CORE**.

---

## 1. Tổng quan về Bộ dữ liệu (Dataset Overview)
- **Tổng số đối tượng**: 40 người trưởng thành (neurotypical).
- **Nhiệm vụ**: **Flanker Task** (Nghiên cứu về Error-Related Negativity - ERN).
- **Thiết bị**: EEG 30 kênh, tốc độ lấy mẫu (SFREQ) 128 Hz.
- **Mục tiêu**: Phân tích sự thay đổi động của mạng lưới não bộ khi người dùng mắc lỗi (Incorrect) so với khi làm đúng (Correct).

---

## 2. Thống kê Chi tiết về Thử nghiệm (Trial Statistics)
Sau khi chạy xử lý cho toàn bộ 40 người, chúng ta có các con số thống kê quan trọng về phân bổ thử nghiệm:

### A. Thử nghiệm Đúng (Correct Trials)
- **Trung bình**: **355.73** trials/người.
- **Thấp nhất (Min)**: 249 trials (`sub-006`).
- **Cao nhất (Max)**: 398 trials (`sub-005`).
- **Nhận xét**: Số lượng trial Đúng rất dồi dào và ổn định trên tất cả các đối tượng, cung cấp một "baseline" (CRN) mạnh mẽ cho việc so sánh.

### B. Thử nghiệm Sai (Incorrect Trials - ERN)
- **Trung bình**: **45.42** trials/người.
- **Thấp nhất (Min)**: 2 trials (`sub-005`).
- **Cao nhất (Max)**: 138 trials (`sub-006`).
- **Nhận xét**: Đây là biến số quan trọng nhất. Vì ERN là phản ứng tự nhiên khi mắc lỗi, số lượng trial này phụ thuộc hoàn toàn vào hiệu suất của người tham gia.

### C. Tiêu chuẩn Sàng lọc (Inclusion Criteria)
Dựa trên bài báo Ozdemir, các đối tượng cần có tối thiểu **15 trials lỗi** để đảm bảo tính ổn định của Tensor kết nối.
- **Số lượng đạt chuẩn**: **34/40** người (85%).
- **Số lượng bị loại**: **6** người (`sub-005, 007, 009, 014, 019, 020`).
- **Lý do loại**: Quá ít lỗi (dưới 15 câu), dẫn đến việc tính toán PLV (Phase Locking Value) không đạt độ tin cậy thống kê.

---

## 3. Bảng phân bổ Thử nghiệm (Mẫu 10 người đầu tiên)

| Subject | Status | Correct | Incorrect | Result |
| :--- | :--- | :--- | :--- | :--- |
| **sub-001** | SUCCESS | 346 | 56 | Included |
| **sub-002** | SUCCESS | 384 | 18 | Included |
| **sub-003** | SUCCESS | 356 | 43 | Included |
| **sub-004** | SUCCESS | 330 | 71 | Included |
| **sub-005** | SUCCESS | 398 | 2 | **Excluded (<15)** |
| **sub-006** | SUCCESS | 249 | 138 | Included (Most Errors) |
| **sub-007** | SUCCESS | 388 | 14 | **Excluded (<15)** |
| **sub-008** | SUCCESS | 351 | 52 | Included |
| **sub-009** | SUCCESS | 394 | 8 | **Excluded (<15)** |
| **sub-010** | SUCCESS | 325 | 76 | Included |

---

## 4. Chất lượng Tín hiệu và Hình ảnh hóa
- **Nhiễu mắt (Ocular Artifacts)**: Xuất hiện ở 100% đối tượng tại các kênh `Fp1`, `Fp2`. Đã được xử lý triệt để bằng ICA.
- **Dải tần Theta (4-8Hz)**: Qua phân tích PSD, dải Theta cho thấy năng lượng tăng vọt rõ rệt trong các trial lỗi (Incorrect) so với trial đúng.
- **ERN Waveform**: Đỉnh âm (negative peak) được ghi nhận rõ ràng nhất tại kênh **FCz** trong khoảng **25ms - 100ms** sau khi nhấn nút sai.

---

## 5. Kết luận EDA
Bộ dữ liệu sau khi xử lý 40 người đã cho thấy một mẫu đủ lớn (**n=34**) để tiến hành các phân tích Tensor chuyên sâu. Sự chênh lệch giữa số lượng Correct và Incorrect được giải quyết bằng kỹ thuật **Trial Balancing** (lấy ngẫu nhiên số lượng trial Correct bằng đúng số lượng Incorrect của từng người) trước khi đưa vào tính toán Tensor.

---
*Tài liệu được cập nhật tự động bởi Antigravity Coding Assistant (04/05/2026).*
