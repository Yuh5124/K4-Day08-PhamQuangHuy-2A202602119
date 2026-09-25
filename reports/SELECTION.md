# Vì sao chọn lô này?

Nguồn số liệu: `outputs/selection_round1.csv` (268 ảnh pool, model cold start `yolov8n` COCO
car+bus+truck, `STRATEGY = "uncertainty"`, `W_U, W_A, W_D = 0.5, 0.3, 0.2`, `MIN_GAP_S = 2.0`) và
contact sheet `outputs/selection_round1.jpg`. Ở vòng 1 chưa có ảnh nào được gán nhãn nên `D = 1.0`
cho mọi ảnh; thứ hạng chỉ do `U` (bất định của 5 box khó nhất) và `A` (số box mơ hồ
0.15 ≤ conf < 0.5, chuẩn hoá theo max = 18) quyết định. Không có ảnh nào `empty = True`, nghĩa là
model luôn dự đoán ít nhất một box, nên không ảnh nào được cộng `EMPTY_BONUS`.

## Top 5 nếu chỉ đủ công rà năm ảnh

Nhận xét chung về 50 dòng đầu: điểm chỉ trải từ 0.9591 (rank 1) xuống 0.8203 (rank 50), mỗi ảnh
có 24–53 box ở conf ≥ 0.05 (trung bình 33.8). Top 50 dồn thành cụm theo thời gian: 10 ảnh trong
khoảng 120–140 s, 9 ảnh trong khoảng 140–157 s, 10 ảnh trong khoảng 40–60 s. Vì camera cố định và
một chiếc xe ở lại trong khung hình vài giây, các ảnh trong cùng cụm gần như trùng cảnh. Nếu lấy
nguyên top 5 theo điểm, lô sẽ có hai cặp gần trùng. Mình đổi hai ảnh để trải lô theo thời gian:

| Thứ tự ưu tiên | Frame | Rank | Score | U | A | t (s) | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 0.9182 | 1.0000 | 72.8 | Điểm cao nhất; `A = 1.0` (18 box mơ hồ, nhiều nhất pool). Đây là đoạn giữa video, chưa có ảnh nào khác trong top 5. |
| 2 | frame_0369.jpg | 2 | 0.9324 | 0.9315 | 0.8889 | 147.6 | Đại diện cụm 140–157 s. Chọn ảnh này thay cho frame_0368 (rank 9, cách 0.4 s) và frame_0372 (rank 6, cách 1.2 s) vì ba ảnh này gần như là một cảnh. |
| 3 | frame_0326.jpg | 4 | 0.9155 | 0.9310 | 0.8333 | 130.4 | Đại diện cụm 120–140 s, đường đông (39 box). Không lấy frame_0331 (rank 5, score 0.9154) vì chỉ cách 2.0 s: vừa đủ `MIN_GAP_S` nhưng vẫn là cùng đoàn xe. |
| 4 | frame_0099.jpg | 8 | 0.9063 | 0.9460 | 0.7778 | 39.6 | Thay frame_0380 (rank 3, t = 152.0 s, chỉ cách frame_0369 4.4 s). U của frame_0099 (0.9460) cao thứ hai trong top 15, chỉ sau frame_0392 và nó phủ đoạn đầu video (0–60 s), phần mà top 5 theo điểm bỏ trống. |
| 5 | frame_0227.jpg | 11 | 0.8915 | 0.9164 | 0.7778 | 90.8 | Lấp khoảng trống 72.8–130.4 s. Ảnh có xe tải lớn ở làn trái; model gợi ý một box sai, gộp nóc xe tải với xe bên cạnh (xem mục 2). Ảnh có 37 box ở conf ≥ 0.05, ít hơn frame_0331 (47 box) nên rà nhanh hơn. |

Lô năm ảnh này trải từ 39.6 s đến 147.6 s, và hai ảnh bất kỳ trong lô cách nhau ít nhất 17 s.
Điểm trung bình của lô là 0.921, so với 0.928 nếu lấy nguyên top 5 theo điểm; phần chênh 0.007 đó nhỏ hơn
nhiều so với giá trị có thêm năm cảnh giao thông khác nhau. Chi phí rà được tính theo số box: ở
lô 12 ảnh, nhãn gợi ý (conf ≥ 0.25) có 169 box nhưng nhãn cuối có 335 box
(`outputs/round1_diff.md`). Như vậy mỗi ảnh cần vẽ thêm khoảng 14.5 box, và công thực tế nằm ở phần
thêm box chứ không nằm ở phần duyệt box gợi ý.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg**: rank 1, score 0.9591, `U = 0.9182`, `A = 1.0`, `n_ambiguous = 18`,
  `n_boxes = 28`. Trên contact sheet đây là ảnh 72.8 s: nhiều xe đi ngược chiều bật đèn pha ở
  làn trái và nhiều xe nhỏ ở xa. Diff vòng 1 cho thấy model chỉ gợi ý 13 box, mình giữ 12, sửa 1
  và thêm 12. Như vậy độ mơ hồ cao ở đây ứng với bỏ sót thật.
- **frame_0331.jpg**: rank 5, score 0.9154, `U = 0.8308`, `A = 1.0`, `n_boxes = 47` (nhiều box
  nhất trong lô). Ảnh có nhiều box gợi ý chồng lên nhau: model đề xuất 20 box, mình xoá 4 box
  trùng hoặc lệch và thêm 19 box (`round1_diff.json`). Ảnh được chọn chủ yếu nhờ `A` chứ không
  nhờ `U`, vì U ở mức thấp so với lô.
- **frame_0392.jpg**: rank 15, score 0.8874, `U = 0.9747` (cao nhất lô), `A = 0.6667`. Ảnh vào lô
  nhờ `U` gần 1, tức có ít nhất 5 box với conf rất gần 0.5. Trên ảnh có hai xe buýt dài bật đèn
  nội thất ở làn phải. Model gợi ý một box chỉ phủ nửa xe buýt; box này đã được kéo rộng (IoU với
  gợi ý 0.68, tính là `edited`).

## Một frame điểm cao nhưng không được chọn và một frame nên xem thêm

- **frame_0372.jpg**: rank 6, score 0.9101, cao hơn 7 ảnh trong lô, nhưng không được chọn vì chỉ
  cách frame_0369 1.2 s, dưới `MIN_GAP_S = 2.0`. Các ảnh frame_0368 (rank 9, cách 0.4 s),
  frame_0330 (rank 12, cách frame_0331 0.4 s) và frame_0271 (rank 16, cách frame_0270 0.4 s) bị
  loại vì cùng lý do. Đây là quyết định đúng: gán nhãn frame_0372 tốn khoảng 30 box nữa, trong
  khi phần lớn các xe đó đã có trong frame_0369.
- **frame_0002.jpg**: rank 18, score 0.8658, t = 0.8 s. Ảnh này nên được xem ở vòng sau. Lô 12
  ảnh bắt đầu từ 39.6 s, nên đoạn 0–39 s không có ảnh nào, dù top 50 có 9 ảnh trong đoạn 0–40 s.
  frame_0002 cách ảnh đã chọn gần nhất 38.8 s. Ở vòng 2, số hạng `D` sẽ tự tăng điểm cho ảnh này
  (D chặn ở 10 s, nên D = 1.0). Tuy vậy vòng 1 đã có thể lấy nó nếu phép chọn ưu tiên phủ thời
  gian hơn khoản chênh 0.02 điểm so với ảnh có điểm thấp nhất lô (frame_0392, 0.8874).

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- Score cao chỉ có nghĩa là model **phân vân** (conf gần 0.5). Nó không cho biết model sai ở đâu.
  Lỗi lớn nhất của cold start là **bỏ sót với conf rất thấp** (FN 206 trên 403 box tham chiếu,
  recall 0.489 trong `metrics_round0.json`), và lỗi này hầu như không làm U tăng. Ví dụ, chiếc xe
  lớn ở sát đáy frame_0099 hoàn toàn không có box gợi ý.
- Điểm bất định không chứng minh rằng thêm ảnh đó sẽ làm AP50 trên test tăng. Muốn biết điều đó
  cần so với lô `STRATEGY = "random"` cùng kích thước, và lab chưa chạy phép đối chứng này.
- Vòng 1 có `D = 1.0` cho mọi ảnh, nên thành phần đa dạng chỉ đến từ `MIN_GAP_S`. Ràng buộc 2 s
  vẫn cho phép hai ảnh cách nhau đúng 2.0 s (frame_0326 và frame_0331) vào cùng lô.
