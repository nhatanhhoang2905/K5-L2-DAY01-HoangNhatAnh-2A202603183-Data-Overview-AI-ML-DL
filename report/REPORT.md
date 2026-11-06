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
   - Record này nói về **cả bức ảnh** (Image-level label).  
   - *Vì sao:* Tác vụ phân loại ảnh dự đoán xác suất nhãn cho toàn bộ khung hình, trong dữ liệu trả về không có tọa độ vị trí (`bbox_xyxy`) hay đa giác viền (`polygon_xy`) để định vị bất kỳ vật thể riêng lẻ nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   - Danh sách tên lớp **đi kèm taxonomy cố định của checkpoint** (ở đây là bộ nhãn ImageNet-1K gồm 1000 lớp cố định của checkpoint `yolo11n-cls.pt`), model **không** tự nghĩ ra lớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   - `class_id` là mã số chuẩn hóa để máy tính xử lý logic và huấn luyện không bị ảnh hưởng bởi lỗi chính tả / khoảng trắng.  
   - `class_name` giúp con người đọc hiểu trực quan.  
   - `taxonomy_name` xác định rõ ngữ cảnh và không gian nhãn của bộ dữ liệu (tránh nhầm lẫn khi cùng một tên lớp nhưng thuộc hai taxonomy có định nghĩa khác nhau).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   - Guideline cần nêu rõ quy tắc ưu tiên: chọn vật thể chính nằm ở vị trí trung tâm, chiếm diện tích lớn nhất hoặc là chủ thể hành động chính của bức ảnh. Nếu không thể chọn một đối tượng duy nhất (như cảnh đường phố hỗn hợp), guideline cần yêu cầu gán nhãn bối cảnh chung (street scene/traffic) hoặc chuyển bài toán sang dạng Multi-label / Object Detection.
6. Vì sao không được lấy `score` của model làm ground truth?  
   - Vì `score` (model confidence) chỉ là xác suất toán học mà mô hình tính toán dựa trên các tham số đã học được, mô hình hoàn toàn có thể tự tin cao (score cao) nhưng vẫn đoán sai. Ground truth phải là sự thật khách quan do con người quan sát, xác minh độc lập theo guideline chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name` = `person` ; `score` = `0.91` ; `bbox_xyxy` = `[364.50, 153.20, 482.30, 355.60]` ; `bbox_width` = `117.80` ; `bbox_height` = `202.40`
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   - Hộp chữ nhật bao quanh người đầu bếp: góc trên-trái tại tọa độ pixel $(364.50, 153.20)$, góc dưới-phải tại tọa độ pixel $(482.30, 355.60)$. Chiều rộng của hộp là $117.80$ pixel và chiều cao là $202.40$ pixel. Gốc tọa độ $(0, 0)$ nằm ở góc trên cùng bên trái ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   - Ngưỡng `0.20`: có **17** vật thể được phát hiện (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', ...]`).  
   - Ngưỡng `0.60`: có **6** vật thể được phát hiện (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person']`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   - Khi hạ ngưỡng, reviewer phải rà soát số lượng hộp tăng lên đáng kể (từ 6 lên 17 hộp) và tốn nhiều công sức để loại bỏ các hộp phát hiện nhầm (false positives - nhận diện nhầm chi tiết nền thành vật thể).
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   - "Hộp bounding box phải bao bọc sát khít (tight boundary) lấy toàn bộ phần nhìn thấy của vật thể, không để lọt khoảng trống thừa quá 5 pixel và không được cắt lẹm vào phần thân của đối tượng."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   - Trên ảnh `kitchen`, người ở góc trái mép ảnh bị cắt cụt (chỉ nhìn thấy cánh tay chìa ra) và phần chân dưới của người đầu bếp bị che khuất sau chiếc bàn gỗ.  
   - Trường hợp này cần **Escalation** (hỏi Lab Coach) để thống nhất quy định: có gán nhãn cho người chỉ nhìn thấy một phần cánh tay hay không, và nếu bị che khuất thì hộp box có cần ước lượng vẽ bao trùm cả phần bị che hay chỉ vẽ phần nhìn thấy.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id` = `kitchen-001` ; `class_name` = `person` ; `score` = `0.90` ; số điểm polygon=`polygon_point_count` = `48` ; 3–5 điểm đầu của `polygon_xy` = `[[482.0, 274.0], [482.0, 273.0], [481.0, 272.0], [481.0, 271.0], [480.0, 270.0]]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   - Polygon cung cấp đường biên hình học thực tế chính xác đến từng pixel theo mép ngoài của đối tượng (dáng người, đường cong của tạp dề), loại bỏ hoàn toàn các vùng không gian nền bị lẫn vào như ở hộp chữ nhật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   - `instance_id` dùng để phân biệt và định danh độc lập từng cá thể riêng biệt trong cùng một bức ảnh (ví dụ có 2 người thì người thứ nhất là `kitchen-001`, người thứ hai là `kitchen-002`).  
   - Nó không phải `class_id` vì không đại diện cho chủng loại đối tượng chung; nó cũng không phải `tracking_id` vì không có tác dụng liên kết hay theo dõi đối tượng di chuyển qua chuỗi các khung hình video liên tiếp.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   - "Đa giác mặt nạ phải bám sát mép ngoài của từng cá thể vật thể; tại các đường cong hoặc nếp gấp trang phục, annotator phải chấm đủ mật độ điểm để phản ánh đúng biên dạng thực tế, không vẽ lẹm sang đối tượng bên cạnh hoặc nền."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   - Trên ảnh `kitchen`, chùm chảo và nồi kim loại treo trên xà ngang đè chồng lên nhau, và mép giao giữa mặt bàn bếp gỗ và chiếc bát đặt bên trên bị che một phần.  
   - Dùng **Escalation** đối với các cụm chảo treo chồng chéo, vì ranh giới kim loại tối màu và bóng phản chiếu khiến mắt thường khó xác định viền của từng chiếc chảo riêng biệt, cần Lab Coach hướng dẫn quy ước gán nhãn cụm (cluster) hay bắt buộc tách rời từng chiếc.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất gán cho toàn bộ bức ảnh (Image-level label). | Ảnh `traffic` có cả xe buýt, ô tô, xe máy nhưng model ImageNet-1K chỉ đoán lớp `cab` với score ~0.51, bỏ qua các phương tiện khác. | Đọc kỹ guideline về quy tắc chọn đối tượng ưu tiên để chọn 1 lớp đại diện chuẩn nhất từ taxonomy. | Kiểm tra nhãn gán có đúng taxonomy và có tuân thủ đúng quy tắc ưu tiên trong guideline hay không. |
| Phát hiện vật thể | Tập hợp các hình chữ nhật (Bounding Box kèm nhãn lớp) cho từng vật thể nhìn thấy. | Trên ảnh `kitchen`, người thò tay ở mép trái bị cắt cụt; các chảo treo trên cao chỉ được nhận diện một vài cái, nhiều cái bị bỏ sót. | Vẽ hộp bao khít sát từng đối tượng được nhìn thấy rõ ràng theo đúng tiêu chuẩn pixel. | Soát lỗi bỏ sót vật thể (False Negative), hộp vẽ quá rộng thừa nhiều nền hoặc hộp vẽ sai lớp đối tượng. |
| Instance segmentation | Tập hợp các đa giác khép kín (`polygon_xy`) kèm nhãn lớp cho từng thực thể đối tượng. | Viền đa giác của các chảo và nồi kim loại treo bị lẫn vào nhau do ánh sáng tối và bị che khuất. | Phóng to ảnh và chấm các điểm đa giác bám sát mép viền của từng đối tượng độc lập. | Kiểm tra độ khít của biên mask (không bị lẹm, không thừa pixel nền) và đảm bảo các vật cùng lớp được tách thành các instance riêng biệt. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   - Chỉ sử dụng các ảnh COCO công khai được cung cấp sẵn trong bài lab; tuyệt đối không tải lên dữ liệu cá nhân (khuôn mặt, biển số xe, thông tin định danh PII) hay dữ liệu nội bộ của doanh nghiệp lên môi trường công cộng.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   - Dừng ngay thao tác gán nhãn/xử lý và báo cáo ngay lập tức cho Lab Coach / Mentor phụ trách lớp học.

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
