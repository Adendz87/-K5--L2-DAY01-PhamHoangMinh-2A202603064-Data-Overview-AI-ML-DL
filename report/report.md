# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

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

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?

## 1. Phân loại ảnh

**1. Record hạng 1:**

- `class_id` = 468
- `class_name` = `cab`
- `rank` = 1
- `score` = 0.510915
- `taxonomy_name` = `ImageNet-1K`

**2. Record này nói về cả bức ảnh hay một vật thể riêng trong ảnh? Vì sao?**

Record này mô tả nội dung tổng thể của bức ảnh. Model phân loại toàn bộ ảnh và đưa ra các lớp có khả năng phù hợp nhất, thay vì xác định vị trí của từng vật thể riêng lẻ. Trong trường hợp này, lớp `cab` có điểm dự đoán cao nhất.

**3. Danh sách tên lớp (class list) do đâu mà có?**

Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm với checkpoint model. Model không tự nghĩ ra tên lớp mới mà dự đoán dựa trên các lớp có sẵn trong ImageNet-1K.

**4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**

Vì mỗi thông tin có mục đích riêng:
- `class_id`: Mã số định danh của lớp.
- `class_name`: Tên dễ đọc của lớp.
- `taxonomy_name`: Cho biết lớp thuộc hệ thống phân loại nào.

Ghi đầy đủ giúp tránh nhầm lẫn giữa các hệ thống phân loại khác nhau và thuận tiện cho việc lưu trữ, xử lý, đối chiếu dữ liệu.

**5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**

Guideline cần quy định rõ tiêu chí chọn lớp chính, chẳng hạn ưu tiên chủ thể nổi bật nhất, chiếm diện tích lớn nhất hoặc là nội dung chính của bức ảnh. Đồng thời cần có quy tắc xử lý khi ảnh có nhiều chủ thể quan trọng hoặc không xác định được chủ thể chính.

**6. Vì sao không được lấy `score` của model làm ground truth?**

Vì `score` chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó là chính xác. Ground truth phải được xác định dựa trên nội dung thực tế của ảnh và guideline do con người thống nhất. Vì vậy, dù model có score cao thì vẫn cần được kiểm tra và xác nhận.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?


## 2. Phát hiện vật thể

**1. Một record trong `detection_predictions.json` (sample_id = kitchen):**

- `class_name` = `person`
- `score` = `0.912625`
- `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`
- `bbox_width` = `113.58`
- `bbox_height` = `279.68`

**2. Giải thích bounding box:**

Box bắt đầu tại góc trên bên trái có tọa độ `(385.33, 69.24)` và kết thúc tại góc dưới bên phải có tọa độ `(498.92, 348.92)`. Box bao quanh một người trong ảnh, có chiều rộng khoảng `113.58 pixel` và chiều cao khoảng `279.68 pixel`.

**3. So sánh số lượng prediction theo ngưỡng:**

Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Dữ liệu bạn gửi hiện tại chỉ chứa kết quả với ngưỡng 0.35. Ngưỡng Số prediction 0.20 Chưa có dữ liệu 0.35 11 0.60 Chưa có dữ liệu

**4. Công việc của reviewer khi hạ ngưỡng:**

Khi hạ ngưỡng, mô hình giữ lại nhiều prediction hơn. Reviewer cần kiểm tra thêm các box có confidence thấp để phát hiện false positive, loại bỏ box trùng lặp, kiểm tra nhãn lớp và điều chỉnh những bounding box không chính xác.

**5. Quy tắc box:**

Bounding box phải bao quanh toàn bộ phần vật thể nhìn thấy được, sát với đường viền vật thể nhất có thể và hạn chế tối đa phần nền không thuộc vật thể. Mỗi vật thể riêng biệt cần có một bounding box riêng. Nếu vật thể bị che khuất hoặc cắt khỏi khung hình, chỉ box phần có thể quan sát được.


**6. Vật thể bị che khuất:**

Trên ảnh kitchen, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Trong dữ liệu JSON, có một số box nằm sát mép ảnh, ví dụ box oven có tọa độ x bắt đầu là 0.11 pixel. Tuy nhiên, JSON chưa đủ thông tin để kết luận vật thể bị che khuất hay bị cắt mép. Cần xem ảnh detection_predictions.png để xác nhận.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?



## 3. Phân đoạn theo từng đối tượng

**1. Một record trong `segmentation_predictions.json` (`sample_id = kitchen`):**

Chọn đối tượng `potted plant`:

- `instance_id` = `kitchen-004`
- `class_name` = `potted plant`
- `score` = `0.632235`
- `polygon_point_count` = `100`
- 5 điểm đầu của `polygon_xy` = `[[1.0, 2.0], [1.0, 148.0], [3.0, 148.0], [3.0, 147.0], [4.0, 146.0]]`

**2. So với bounding box, polygon cung cấp thêm thông tin gì?**

Bounding box chỉ biểu diễn vùng hình chữ nhật bao quanh vật thể. Polygon mô tả đường biên thực tế của vật thể bằng nhiều điểm tọa độ, giúp thể hiện hình dạng, các phần lồi lõm và diện tích vật thể chính xác hơn.

**3. Ý nghĩa của `instance_id`:**

`instance_id` là mã định danh riêng cho từng đối tượng được phát hiện trong một ảnh. Nó giúp phân biệt các cá thể khác nhau, kể cả khi chúng có cùng class.

Nó không phải `class_id` vì `class_id` xác định loại đối tượng, ví dụ `person` hoặc `bowl`. Nó cũng không phải tracking ID vì `instance_id` không dùng để theo dõi cùng một đối tượng xuyên suốt nhiều khung hình hoặc video.

**4. Quy tắc vẽ biên mask:**

Vẽ polygon sát theo đường viền của phần vật thể có thể nhìn thấy, không bao gồm vùng nền xung quanh. Nếu vật thể bị che khuất hoặc cắt khỏi khung hình, chỉ vẽ phần thực sự quan sát được, không tự suy đoán phần bị khuất hoặc nằm ngoài ảnh.

**5. Trường hợp biên bị cắt trên ảnh kitchen:**

Vị trí: Chiếc bát màu hồng ở góc dưới bên trái ảnh, nơi bát tiếp xúc với mặt bàn và các vật thể xung quanh.

Lựa chọn: Guideline (áp dụng hướng dẫn).

Lý do: Đây là trường hợp các vật thể nằm sát nhau hoặc bị che khuất một phần. Áp dụng guideline bằng cách chỉ đánh dấu phần đối tượng có thể nhìn thấy, giữ đường biên theo hình dạng thực tế và không tự suy đoán phần bị che khuất. Chỉ escalation khi guideline không đủ để giải quyết trường hợp mơ hồ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 4. Vòng đời và kiểm tra chất lượng

**1. Phân loại ảnh (Image Classification)**

- **Ground truth:** Một nhãn class đại diện cho nội dung chính của toàn bộ ảnh.
- **Lỗi hoặc chỗ khó:** Ảnh traffic có prediction `cab` với score = 0.510915. Score không đồng nghĩa với ground truth.
- **Annotator:** Xác định nội dung chính của ảnh và gán class phù hợp theo taxonomy.
- **Reviewer:** Kiểm tra nhãn có phản ánh đúng nội dung ảnh và đúng taxonomy không.

**2. Phát hiện vật thể (Object Detection)**

- **Ground truth:** Mỗi vật thể có một class label và một bounding box với tọa độ `xyxy`.
- **Lỗi hoặc chỗ khó:** Ảnh kitchen có nhiều vật thể như `person`, `bowl`, `oven`. Một số vật thể bị che khuất hoặc nằm sát mép ảnh nên khó xác định box chính xác.
- **Annotator:** Xác định từng vật thể, gán đúng class và vẽ bounding box bao quanh phần nhìn thấy.
- **Reviewer:** Kiểm tra vật thể bị bỏ sót, class sai, box lệch, box quá rộng hoặc quá hẹp và các box trùng lặp.

**3. Instance Segmentation**

- **Ground truth:** Mỗi đối tượng có class label, instance ID và polygon/mask mô tả vùng pixel thuộc vật thể.
- **Lỗi hoặc chỗ khó:** Chậu cây `potted plant` trong ảnh kitchen bị cắt ở mép trái. Các vật thể trên bàn nằm gần nhau nên khó xác định ranh giới mask.
- **Annotator:** Vẽ polygon sát đường biên phần vật thể nhìn thấy và tách riêng từng đối tượng.
- **Reviewer:** Kiểm tra mask có đúng đường viền không, có lấn sang vật thể khác không, có bỏ sót vùng nhìn thấy không và instance ID có phân biệt đúng từng đối tượng không.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu:**

Không chia sẻ, sao chép hoặc sử dụng dữ liệu của bài lab ra bên ngoài khi chưa được cho phép; chỉ sử dụng dữ liệu đúng mục đích và phạm vi được giao.

**2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi:**

Tôi sẽ dừng xử lý dữ liệu đó và báo cho Lab Coach hoặc người phụ trách để được hướng dẫn.

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
