# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python: 3.13.18 / PyTorch: 2.11.0+cpu / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id` = `468` ; `class_name` = `cab` ; `rank` = 1 ; `score` = `0.510915` ; `taxonomy_name` = `ImageNet-1K`
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   - Record này nói về **toàn bộ bức ảnh**, chứ không nói riêng về một đồ vật nào.  
   - *Vì sao:* Mô hình chỉ nhìn tổng thể bức ảnh rồi đưa ra một phỏng đoán chung nhất (ở đây là xe taxi - `cab`). Trong kết quả trả về hoàn toàn không có tọa độ khung viền (`bbox_xyxy`) hay vị trí cụ thể nào để chỉ ra một chiếc xe riêng lẻ nằm ở đâu.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   - Danh sách tên lớp **có sẵn từ trước, đi kèm với mô hình** (ở đây là danh mục 1.000 lớp của bộ ImageNet-1K gắn liền với checkpoint `yolo11n-cls.pt`). Mô hình **không tự sáng chế** ra tên lớp mà chỉ chọn từ danh mục đã học.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   - `class_id`: Là mã số để máy tính hiểu chính xác, không lo bị nhầm do gõ sai chính tả hay dấu cách.  
   - `class_name`: Là tên chữ (như `cab`) để con người đọc vào hiểu được ngay.  
   - `taxonomy_name`: Cho biết tên này lấy từ từ điển/danh mục nào (ví dụ ImageNet-1K hay COCO). Điều này giúp tránh nhầm lẫn khi hai danh mục khác nhau dùng cùng một từ nhưng lại quy định nghĩa khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   - Bản hướng dẫn (guideline) cần quy định rõ thứ tự ưu tiên: ví dụ chọn đồ vật to nhất, nằm ngay giữa ảnh, hoặc đối tượng nổi bật nhất đập vào mắt người xem. Nếu ảnh quá lộn xộn (như cảnh đường phố hỗn hợp nhiều xe), quy tắc phải nói rõ nên chọn một nhãn bao quát (như đường phố/giao thông) hay chuyển sang dạng bài toán khoanh từng vật thể riêng.
6. Vì sao không được lấy `score` của model làm ground truth?  
   - Vì `score` (độ tự tin) chỉ là con số thể hiện mô hình "nghĩ" là mình đoán đúng đến mức nào. Mô hình có thể rất tự tin (điểm cao) nhưng thực tế vẫn đoán hoàn toàn sai. Nhãn chuẩn (ground truth) bắt buộc phải do mắt người trực tiếp quan sát và xác nhận cẩn thận theo quy tắc.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name` = `person` ; `score` = `0.91` ; `bbox_xyxy` = `[364.50, 153.20, 482.30, 355.60]` ; `bbox_width` = `117.80` ; `bbox_height` = `202.40`
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   - Khung chữ nhật khoanh người đầu bếp: góc trên-trái nằm tại điểm $(364.50, 153.20)$, góc dưới-phải nằm tại điểm $(482.30, 355.60)$. Khung này rộng $117.80$ pixel và cao $202.40$ pixel. Điểm gốc $(0, 0)$ được tính từ góc trên cùng bên trái của bức ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   - Ngưỡng dễ tính `0.20`: mô hình tìm thấy **17** đồ vật/người (chấp nhận cả các phỏng đoán còn ngập ngừng, chưa chắc chắn).  
   - Ngưỡng khắt khe `0.60`: mô hình chỉ giữ lại **6** đồ vật/người (chỉ lấy những trường hợp mô hình rất tự tin).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   - Khi hạ ngưỡng, số hộp mô hình đoán tăng vọt (từ 6 lên 17). Người kiểm tra (reviewer) sẽ phải mất thêm nhiều công sức rà soát và xóa bớt những hộp đoán ẩu, đoán sai (những chỗ mô hình nhìn một mảng tường, cái bóng hay nền nhà mà lại tưởng nhầm là đồ vật).
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   - "Khung hộp chữ nhật phải ôm vừa khít toàn bộ phần nhìn thấy của đồ vật: không khoanh lẹm vào thân đồ vật, và cũng không để thừa khoảng trống ra ngoài nền xung quanh quá 5 pixel."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   - Trên ảnh `kitchen`, người đứng ở mép trái bị cắt cụt (chỉ thấy mỗi cánh tay thò ra), và chân người đầu bếp bị che khuất sau chiếc bàn gỗ.  
   - Chỗ này cần **đưa lên hỏi Lab Coach (escalation)** để thống nhất quy ước chung: người chỉ thấy mỗi cánh tay thì có khoanh không, và đồ vật bị che thì vẽ khung ôm trọn phần bị giấu hay chỉ khoanh phần mắt nhìn thấy.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id` = `kitchen-001` ; `class_name` = `person` ; `score` = `0.90` ; số điểm polygon=`polygon_point_count` = `48` ; 3–5 điểm đầu của `polygon_xy` = `[[482.0, 274.0], [482.0, 273.0], [481.0, 272.0], [481.0, 271.0], [480.0, 270.0]]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   - Khung hộp chữ nhật chỉ cho biết vùng chữ nhật bao quanh, còn đa giác (polygon) vẽ uốn lượn sát theo mép thật của đồ vật (như đường cong cơ thể, vạt tạp dề). Nhờ đó, ta lấy được đúng hình dáng thực tế mà không bị lẫn mảng nền xung quanh vào.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   - `instance_id` dùng để đánh số và phân biệt từng đối tượng riêng lẻ trong cùng một ảnh (ví dụ trong bếp có 2 người thì người thứ nhất mang mã `kitchen-001`, người thứ hai là `kitchen-002`).  
   - Nó không phải `class_id` vì cả 2 người đều chung một loại là `person` (người).  
   - Nó cũng không phải `tracking_id` vì nó chỉ có giá trị trên bức ảnh tĩnh này, không dùng để theo dõi người đó đang đi đâu trong một đoạn clip dài.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   - "Đường viền đa giác phải bám sát mép ngoài của đồ vật. Ở các đoạn uốn cong hoặc nếp gấp quần áo, người gán nhãn cần chấm nhiều điểm để viền ôm thật mịn theo dáng thật, không chấm vội vàng làm lẹm vào người hoặc bao thừa mảng nền."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   - Trên ảnh `kitchen`, chùm xoong chảo kim loại màu tối treo trên giá đè chồng chéo lên nhau, ranh giới bị bóng nhòa rất khó phân biệt viền của từng chiếc.  
   - Chỗ này cần **đưa lên hỏi Lab Coach (escalation)** để xin quy định: chùm chảo đè nhau này sẽ khoanh chung thành một chùm hay bắt buộc phải ráng tách riêng từng chiếc chảo một.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một tên gọi gán chung cho toàn bộ bức ảnh (ví dụ: ảnh xe taxi). | Ảnh `traffic` có cả xe buýt, ô tô, người đi lại nhưng mô hình chỉ đoán nhãn `cab` với độ tin cậy ~0.51, bỏ qua các phương tiện khác. | Đọc kỹ quy tắc ưu tiên để chọn đúng một tên gọi đại diện chuẩn nhất cho bức ảnh. | Soát xem nhãn được chọn có nằm đúng trong danh mục quy định và có đúng đối tượng chính yếu nhất hay không. |
| Phát hiện vật thể | Các khung hộp hình chữ nhật khoanh quanh từng đồ vật nhìn thấy, mỗi hộp kèm theo tên gọi. | Trên ảnh `kitchen`, người đứng mép trái chỉ thò một cánh tay ra; nhiều chiếc chảo treo trên cao bị mô hình bỏ sót hoàn toàn. | Cầm chuột khoanh khung ôm vừa vặn từng đồ vật nhìn thấy rõ ràng theo đúng chuẩn. | Rà soát xem có bị sót đồ vật nào không, khung khoanh có bị rộng quá hay cắt lẹm vào đồ vật không. |
| Instance segmentation | Các đường viền đa giác khép kín ôm sát hình dáng của từng đồ vật riêng lẻ (kèm tên gọi). | Viền của các xoong chảo kim loại treo trên cao bị dính chùm vào nhau do bị bóng và ánh sáng tối. | Phóng to ảnh, chấm cẩn thận từng điểm men theo mép đồ vật để tạo thành viền ôm trọn đối tượng. | Kiểm tra xem viền có bám sát mép thật không (có bị ăn lẹm hay thừa nền không) và các đồ vật đứng gần đã được tách riêng rẽ chưa. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   - Chỉ dùng đúng các bức ảnh mẫu có sẵn của bài lab; tuyệt đối không tải ảnh cá nhân, mặt người quen, biển số xe hay dữ liệu riêng tư của công ty lên Colab hoặc GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   - Dừng ngay công việc và báo ngay cho Lab Coach hoặc người hướng dẫn lớp học.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
