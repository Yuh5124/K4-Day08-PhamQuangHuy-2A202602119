# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Phạm Quang Huy

Công cụ gán nhãn đã dùng: CVAT Docker local (v2.76.0). Pre-label được import và nhãn được export
theo định dạng Ultralytics YOLO Detection 1.0, rồi đóng gói bằng `tools/pack_labels.py`.

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`,
`outputs/metrics_round1.json`, `outputs/selection_round1.csv`, `outputs/round1_diff.md/.json`.
Nhãn test do mô hình tạo và chưa được người rà, nên mọi số AP50/P/R dưới đây chỉ là **mức khớp với
bộ tham chiếu đó**, không phải độ đúng tuyệt đối.

## 1. Dữ liệu và cách chia tập

Video quay bằng camera cố định trên cầu vượt, không có lần cắt cảnh nào (điểm đổi cảnh cao nhất
0.08, `data/DATA.md`). Ảnh được trích 2.5 frame/giây, nên hai frame liền nhau chỉ cách 0.4 s và
gần như giống hệt nhau. Một chiếc xe ở lại trong khung hình vài giây. Nếu chia ngẫu nhiên, cùng một
chiếc xe ở cùng một vị trí sẽ xuất hiện ở cả tập huấn luyện lẫn tập test. Khi đó mô hình được chấm
trên chính những xe nó đã học.

Vì vậy dữ liệu được chia theo trục thời gian. Tập test gồm 4 đoạn, có tâm ở 20/60/100/140 s, mỗi
đoạn 5 ảnh. Quanh mỗi đoạn có vùng đệm ±4 s (112 ảnh bị loại). Pool còn 268 ảnh, và ảnh pool gần
test nhất vẫn cách 4.4 s. Nếu chia ngẫu nhiên, số đo test sẽ bị **lệch lên (lạc quan)**: AP50 và
recall cao hơn khả năng thật trên cảnh mới, vì đó là rò rỉ dữ liệu (data leakage). Chênh lệch giữa
các vòng cũng sẽ bị phóng đại, vì mỗi ảnh train thêm gần như "chép" luôn một ảnh test lân cận.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `reports/rounds_table.md` (403 box tham chiếu được chấm, 14 box cao dưới 16 px bị bỏ qua):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Ở conf 0.25, cold start có TP 197, FP 16, FN 206 (`metrics_round0.json`). Precision cao (0.925)
nhưng recall chỉ 0.489: mô hình ít báo nhầm nhưng **bỏ sót hơn nửa số xe**. Tuy vậy AP50 vẫn đạt
0.771, vì AP50 tính trên mọi mức conf. Nhiều xe vẫn được phát hiện nhưng với conf dưới 0.25.

Trên `outputs/compare_round0.jpg`, các box vàng (FN) tập trung ở ba nhóm:

- **Xe gần camera đi ngược chiều, đèn pha chói.** Có xe con lớn ở góc dưới trái frame_0050, hai xe
  lớn ở hàng dưới frame_0250 và hai xe ở hàng dưới frame_0350. Thân xe tối, hai đèn pha cháy sáng
  và vệt sáng lớn trên mặt đường, khác hẳn ảnh xe ban ngày trong COCO. Đó là lý do recall của xe
  large chỉ 0.561, dù đây là những xe dễ thấy nhất với người.
- **Xe nhỏ ở xa, chỉ còn cụm đèn.** Có ở cụm làn trái, gần chân trời, trong frame_0150 và
  frame_0350. Recall small chỉ **0.182** (66 box tham chiếu), thấp nhất trong ba nhóm kích thước.
- **Xe ở mép ảnh hoặc bị nhoè.** Ví dụ xe cắt ở mép phải frame_0250.

Recall tăng dần từ small (0.182) lên medium (0.547) và large (0.561). Vậy xe nhỏ là điểm yếu lớn
nhất. Nhưng việc recall của xe large không cao hơn medium bao nhiêu cho thấy lỗi của cold start
không chỉ do kích thước: điều kiện ánh sáng ban đêm cũng làm mô hình bỏ sót.

**Ca cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:** ở frame_0150, làn phải
(khoảng x 470–520, y 540–580 trên panel), tham chiếu có hai box chồng nhau cho cụm đèn hậu đỏ, còn
cold start vẽ một box lớn bao cả cụm và bị tính FP (đỏ) cộng FN (vàng). Có thể đó là hai xe sát
nhau, cũng có thể là một xe cộng ánh đèn phản chiếu. Tham chiếu do mô hình tạo nên có thể sai ở
đây, và phải xem ảnh gốc mới kết luận được. Tương tự, box FP đỏ ở làn trái frame_0350 (khoảng
x 760–825, y 1250–1290 trên grid) nằm ở vùng có đèn phản chiếu trên mặt đường. Nếu tham chiếu bỏ
sót một xe thật ở đó, đấy là lỗi của tham chiếu chứ không phải của mô hình.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D`, với trọng số 0.5 / 0.3 / 0.2:

- **U (bất định):** mỗi box có `u = 1 − |2·conf − 1|`, đạt 1 khi conf = 0.5, tức lúc mô hình phân
  vân nhất. U là trung bình 5 giá trị u lớn nhất trong ảnh, nghĩa là ảnh chứa vài box mà mô hình
  "không chắc" nhất.
- **A (mơ hồ):** số box có 0.15 ≤ conf < 0.5, chia cho số lớn nhất trong pool (18). Ảnh có nhiều
  box nằm quanh ngưỡng thì A cao.
- **D (đa dạng):** khoảng cách thời gian tới ảnh đã gán gần nhất, chặn ở 10 s. Ở vòng 1 chưa có
  nhãn nên D = 1 cho mọi ảnh.
- **MIN_GAP_S = 2.0:** khi chọn tham lam theo score, bỏ qua ảnh cách một ảnh đã chọn dưới 2 s. Lý
  do là camera đứng yên, nên hai ảnh sát nhau gần như trùng: gán cả hai tốn gấp đôi công mà mô
  hình học thêm rất ít.

Dẫn chứng từ `reports/SELECTION.md` và `outputs/selection_round1.csv`:

- **frame_0182** (rank 1, score 0.9591, A = 1.0) được chọn chủ yếu nhờ có nhiều box mơ hồ nhất.
  Khi rà, mô hình chỉ gợi ý 13 box, mình thêm 12 box.
- **frame_0331** (rank 5, U = 0.8308, A = 1.0) có 47 box ở conf ≥ 0.05. Sau khi rà, 4 box gợi ý bị
  xoá vì trùng hoặc lệch, và 19 box được thêm. Đây là ảnh tốn công nhất lô.
- **frame_0392** (rank 15, U = 0.9747, cao nhất lô) vào lô nhờ U, dù A chỉ 0.667. Box xe buýt
  gợi ý chỉ phủ nửa thân xe.
- **Frame khác: frame_0372** (rank 6, score 0.9101) không được chọn vì chỉ cách frame_0369 1.2 s.
  Tương tự, frame_0368, frame_0330 và frame_0271 đều cách một ảnh trong lô 0.4 s. MIN_GAP_S đã
  tiết kiệm được công gán khoảng 30–50 box mỗi ảnh trùng. Ngược lại, **frame_0002** (rank 18,
  t = 0.8 s) nên được xem ở vòng sau, vì cả lô không có ảnh nào trước 39.6 s.

Với ngân sách chỉ 5 ảnh, mình chọn frame_0182, frame_0369, frame_0326, frame_0099 và frame_0227
thay cho nguyên top 5 theo điểm. Top 5 theo điểm có hai cặp gần trùng: frame_0326/frame_0331 cách
2.0 s, frame_0369/frame_0380 cách 4.4 s.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Không.** Score cao chỉ nói mô
hình đang phân vân (conf ≈ 0.5), không nói box đó sai. Lỗi lớn nhất của cold start là bỏ sót với
conf rất thấp, ví dụ chiếc xe lớn sát đáy frame_0099 không có box gợi ý nào. Loại lỗi này không
làm U tăng. Hơn nữa, ảnh bất định có thể chỉ bất định vì nó mơ hồ thật với cả người gán (xe rất
xa, ánh phản chiếu). Khi đó thêm nhãn cũng khó giúp mô hình. Muốn chứng minh cần so với lô
`STRATEGY = "random"` cùng kích thước trên cùng tập test.

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 335 | 0.617 | -0.155 | 1.000 | 0.149 | 0.259 | 0.000 | 0.135 | 0.488 |

**Mức sửa pre-label vòng 1** (`outputs/round1_diff.md`): 12 ảnh, mô hình đề xuất 169 box
(conf ≥ 0.25), sau khi sửa còn 335 box.

| accepted | edited | deleted (FP của model) | added (FN của model) | accept rate |
| ---: | ---: | ---: | ---: | ---: |
| 154 | 7 | 8 | 174 | 91% |

Accept rate 91% nghĩa là box nào mô hình đã vẽ thì phần lớn đúng. Đây cũng là precision cao của
cold start nhìn từ phía nhãn. Nhưng số box **thêm** (174) còn nhiều hơn số box giữ lại (154), nên pre-label chỉ
phủ chưa tới một nửa số xe. Kết quả này khớp với recall 0.489 trên test. Lỗi chính của nhãn AI ban
đầu là **bỏ sót**, còn box sai hoặc trùng chỉ có 8.

**AP50 thay đổi:** AP50 giảm từ 0.771 (cold start) xuống 0.617 ở vòng 1 (Δ −0.155). Precision
tăng lên 1.000 (FP 16 → 0) nhưng recall giảm mạnh từ 0.489 xuống 0.149 (FN 206 → 343), F1 từ
0.640 xuống 0.259. Nhóm bị ảnh hưởng nặng nhất là **xe ở xa (small)**: recall 0.182 → 0.000, tức
ở conf 0.25 mô hình không còn phát hiện được xe nhỏ nào. Medium giảm 0.547 → 0.135, large giảm ít
hơn 0.561 → 0.488.

**Ca kết quả đổi sau fine-tune** (so `compare_round0.jpg` với `compare_round1.jpg`): ở cả 4 frame
test, mô hình vòng 1 chỉ còn 3 TP mỗi frame, 0 FP, và FN tăng ở mọi frame (0050: 7 → 15, 0150:
10 → 17, 0250: 9 → 12, 0350: 14 → 20).

- *Tốt lên:* xe lớn đèn pha chói ở góc dưới trái frame_0250 là FN (vàng) ở cold start, sang vòng 1
  thành TP (xanh). Xe lớn ở dưới bên phải frame_0350 cũng đổi từ vàng sang xanh. Lô sửa vòng 1 có
  nhiều ca cùng kiểu, ví dụ frame_0099 đã thêm xe sát đáy ảnh (`round1_diff.json`: 12 added).
- *Xấu đi:* các xe nhỏ ở xa phía trên bên phải frame_0050 là TP (xanh) ở cold start, sang vòng 1
  đều thành FN (vàng); xe ở giữa ảnh, hàng thứ hai từ dưới lên, cũng đổi từ xanh sang vàng. Điều
  này khớp với recall small giảm từ 0.182 xuống 0.000.

**Phân biệt ba nguồn bằng chứng:**

- *Quan sát độc lập* (`BLIND_SCAN.md`, đã khoá bằng `blind_lock.json` trước khi mở pre-label):
  frame_0099, mình đếm được 25 xe và dự đoán AI sẽ bỏ sót xe bị che khuất và xe ở xa chỉ còn đèn.
- *Lỗi pre-label đã sửa* (`REVIEW_LOG.csv` và `round1_diff.md`): pre-label của frame_0099 chỉ có
  13 box, và nhãn cuối có 25 box (12 added), khớp đúng số xe mình đếm lúc quét mù. Các box thêm
  gồm xe ở xa gần chân trời bên trái, đúng dự đoán trong quét mù. Có cả một loại mình **không**
  dự đoán: xe lớn đi ngược chiều sát đáy ảnh với đèn pha chói cũng bị bỏ sót. Các lỗi khác: box
  trùng hoặc lệch ở frame_0331 (xoá 4), box gộp hai xe ở frame_0227 (xoá, vẽ lại thành hai box),
  box xe buýt chỉ phủ nửa thân ở frame_0392 (edited).
- *Kết quả mô hình sau train* (`metrics_round1.json`, `compare_round1.jpg`): chỉ lấy từ số đo
  trên 20 ảnh test, không lấy bảng mAP50 Ultralytics in ra lúc train (bảng đó tính trên chính ảnh
  train).

**Ca khó theo guideline:** xe đi ngược chiều ở xa, thân tối, chỉ thấy hai đèn pha kèm vệt sáng
dài trên mặt đường (làn trái frame_0326 và frame_0380). Cách xử lý mình chọn và giữ cho mọi ảnh:
vẽ box theo **thân xe đoán được quanh cụm đèn**, không khoanh riêng hai chấm đèn, và **không** kéo
box xuống vệt sáng trên mặt đường. Xe cao dưới khoảng 16 px (sát chân trời) thì gán khi còn thấy
rõ cặp đèn tách khỏi xe bên cạnh. Nếu hai cụm đèn dính vào nhau thì không gán, vì loại box này bị
bỏ qua khi chấm (`det_eval.py`, `MIN_BOX_H_PX = 16`). Xe buýt và xe tải dài được vẽ một box phủ
toàn bộ phần nhìn thấy, không cắt theo khung cửa sổ sáng.

## 5. Kết luận và giới hạn

**So với cold start:** AP50 0.771 → 0.617 (Δ −0.155). Mức giảm lớn hơn nhiều so với ngưỡng nhiễu
0.01 (`DATA.md`), nên mô hình sau fine-tune vòng 1 **kém hơn** cold start trên tập test này.

**Dừng hay tiếp tục:** **Dừng.** Δ AP50 = −0.155 và recall giảm ở cả ba nhóm kích thước, nên theo
nguyên tắc dưới đây mình không làm tiếp vòng 2 với cùng chiến lược. Nguyên tắc mình dùng:

- Tiếp tục vòng 2 nếu recall (nhất là small/large) tăng rõ mà precision không giảm mạnh. Khi đó
  thêm nhãn đang có ích.
- Dừng hoặc đổi chiến lược (thử `random` làm đối chứng) nếu Δ AP50 < 0.01. Lúc đó mình sẽ ưu tiên
  sửa chất lượng nhãn (xem tự QC bên dưới) hơn là thêm ảnh.

**Hai ca còn yếu hoặc bất định đề xuất cho vòng sau:**

1. **Xe nhỏ ở xa gần chân trời** (recall small 0.182 ở cold start). Ví dụ là cụm xe làn trái trong
   frame_0002 (rank 18, t = 0.8 s), thuộc đoạn đầu video mà lô vòng 1 chưa phủ (lô bắt đầu từ
   39.6 s). Chi phí cao: mỗi ảnh có khoảng 10–15 xe nhỏ phải vẽ tay, và nhiều xe dưới 16 px bị bỏ
   qua khi chấm, nên lợi ích trên AP50 có giới hạn.
2. **Xe lớn gần camera bị đèn pha chói hoặc cắt ở mép dưới** (recall large 0.561). Ví dụ là
   frame_0372 (rank 6), bị loại ở vòng 1 vì gần frame_0369. Ở vòng 2, D sẽ phạt ảnh này (cách
   frame_0369 chỉ 1.2 s, D = 0.12), nên nếu cần ca kiểu này thì nên chọn một ảnh đoạn 0–35 s
   thay vì frame_0372. Nguy cơ gần trùng: chọn thêm ảnh trong cụm 147–150 s gần như lặp lại cùng
   đoàn xe đã gán.

**Chi phí gán nhãn:** vòng 1 phải thêm 174 box tay trên 12 ảnh (khoảng 14.5 box mỗi ảnh). Công thực
tế nằm ở thêm box chứ không ở duyệt box gợi ý. Nếu mô hình vòng 1 tăng recall, pre-label vòng 2
sẽ phủ nhiều hơn và công sẽ giảm. Đó cũng là một tiêu chí để quyết định có làm tiếp hay không.

**Giới hạn của kết luận:**

- **Tập test nhỏ:** 20 ảnh, 403 box được chấm, lại lấy từ 4 đoạn 5 ảnh gần nhau, nên thực chất chỉ
  có khoảng 4 cảnh. Chênh dưới khoảng 0.01 AP50 là nhiễu. Một ảnh test khó có thể làm lệch cả kết
  quả.
- **Nhãn tham chiếu do mô hình tạo, chưa được rà:** AP50 đo mức "giống mô hình tham chiếu". Nếu mô
  hình fine-tune học đúng theo guideline mà khác thói quen của tham chiếu, ví dụ box xe buýt dài
  hơn hoặc gán thêm xe xa, thì điều đó có thể bị tính FP/FN oan (xem ca frame_0150 ở mục 2).
- **Luật bỏ qua box < 16 px:** loại 14 box xa nhất khỏi phép chấm. Recall small đo trên 66 box còn
  lại, và cải thiện ở xe rất xa sẽ không hiện trong số đo.
- **Ảnh gần trùng trong train:** 12 ảnh nhưng có cặp chỉ cách 2.0 s (frame_0326/frame_0331), nên
  lượng thông tin thật ít hơn 12 cảnh độc lập.

**Nếu AP50 giảm, mình sẽ kiểm tra trước khi train thêm:**

1. Nhãn vòng 1 có bỏ sót xe rõ không. Một ảnh thiếu nhiều xe sẽ dạy mô hình rằng xe là nền.
2. Box có nhất quán không, ví dụ có box nào ôm vệt đèn trên mặt đường, hoặc gộp hai xe thành một.
3. Test label có bị đổi không (`label_hashes.json`, `check_submission.py`) và ảnh test có lọt vào
   `labels/round1/` không.
4. Đọc `compare_round1.jpg` xem FP mới nằm ở ánh phản chiếu, hay ở chỗ tham chiếu có thể sai.

**Tự QC nhãn vòng 1:** mình vẽ đè nhãn cuối lên ảnh cho cả 12 ảnh, rồi đối chiếu với
`round1_diff.json` và `REVIEW_LOG.csv`. Class id đều là `0`. Các ca phát hiện khi QC:
frame_0369 (thiếu nhiều xe ở giữa ảnh), frame_0270 (thiếu xe tải lớn giữa làn), frame_0312 (thiếu
xe tải thùng trắng và một box gộp nhiều xe), frame_0331 (thiếu xe tối ở góc dưới trái). Mình đã
sửa frame_0369 và export lại (18 → 38 box, thêm 24 box so với pre-label); mô hình vòng 1 trong bảng
trên được train với bản nhãn đã sửa này (335 box). frame_0270, frame_0312 và frame_0331 **chưa
sửa**, còn để lại cho lần sau; đây cũng là một nguyên nhân có thể khiến mô hình học rằng một số xe
là nền.
