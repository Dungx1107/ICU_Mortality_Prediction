# Quy trình xử lý dữ liệu — ICU Mortality Prediction

Tài liệu ghi lại toàn bộ quá trình từ dữ liệu eICU Demo v2.0.1 thô đến bảng
`ai_ready.csv` phục vụ huấn luyện mô hình XGBoost dự đoán tử vong ICU.

---

## 1. Nguồn dữ liệu

Bộ dữ liệu: **eICU Demo v2.0.1** (phiên bản demo công khai của eICU-CRD).

Bốn bảng gốc được sử dụng:

| Bảng | Nội dung | Vai trò |
|------|----------|---------|
| `patient.csv` | Nhân khẩu học, trạng thái xuất viện | Nguồn nhãn `mortality`, age, giới tính, cân nặng, chiều cao |
| `apacheApsVar.csv` | Chỉ số sinh lý APACHE 24h đầu | Phục dựng điểm GCS (eyes, motor, verbal) |
| `vitalPeriodic.csv` | Sinh hiệu đo tự động 5 phút/lần | 9 đặc trưng sinh hiệu trong 24h đầu |
| `lab.csv` | Kết quả xét nghiệm máu | 10 đặc trưng xét nghiệm trong 24h đầu |

Khóa chính xuyên suốt: **`patientunitstayid`** (mã đợt nằm ICU).

---

## 2. Tiêu chí lọc cohort (Cohort Filtering)

Áp dụng thống nhất cho toàn bộ pipeline, theo phương pháp bài báo gốc:

1. **Tuổi ≥ 18** — loại bỏ nhi khoa (sinh lý trẻ em khác người lớn).
2. **Thời gian nằm ICU ≥ 240 phút (4 giờ)** — đảm bảo đủ cửa sổ thu thập dữ liệu.
3. **Có nhãn `hospitaldischargestatus`** — không thiếu nhãn huấn luyện.

Kết quả: **2.326 bệnh nhân** trong cohort cuối cùng (từ 2.520 ban đầu).

---

## 3. Xử lý từng bảng

### 3.1. `patient.csv` → `patient_clean.csv`

**Cột giữ lại:** `patientunitstayid`, `age`, `gender`, `admissionweight`,
`admissionheight`, `unitdischargeoffset`, `hospitaldischargestatus`.

**Xử lý:**
- `age`: thay `"> 89"` → `90` (giá trị HIPAA), ép kiểu số.
- Áp dụng cohort filtering (mục 2).
- **Plausibility filter:**
  - `admissionweight`: chỉ giữ `[20, 300]` kg.
  - `admissionheight`: chỉ giữ `[100, 230]` cm.
  - Giá trị ngoài khoảng → `NaN` (không xóa dòng).
- **Feature engineering:**
  - `bmi = weight / (height/100)²`, chỉ tính khi cả hai hợp lệ.
  - BMI ngoài `[10, 80]` → `NaN`.
  - `male_sex`: `1 = Male`, `0 = Female`, `NaN = Unknown`.
  - `mortality`: `1 = Expired`, `0 = Alive`.

**Kết quả:** 2.326 dòng × 5 cột (`patientunitstayid`, `age`, `male_sex`, `bmi`, `mortality`).
BMI mean = 28.57 ± 8.02 — khớp với bài báo gốc (28.6 ± 8.1).

---

### 3.2. `apacheApsVar.csv` → `gcs_clean.csv`

**Cột giữ lại:** `patientunitstayid`, `eyes`, `motor`, `verbal`.

**Xử lý:**
- Giá trị `< 0` (sentinel `-1` = không đánh giá được) → `NaN`.
- Kiểm tra phạm vi hợp lệ: eyes `[1,4]`, motor `[1,6]`, verbal `[1,5]`.
- Phục dựng: `gcs = eyes + motor + verbal`.
  - Nếu bất kỳ thành phần nào `NaN` → `gcs = NaN`.
- GCS hợp lệ nằm trong `[3, 15]`.
- Lọc theo cohort (chỉ giữ bệnh nhân trong `patient_clean`).

**Kết quả:** 2.074 dòng × 2 cột (`patientunitstayid`, `gcs`).
Coverage: 89.2% cohort có GCS.

---

### 3.3. `vitalPeriodic.csv` → `vital_clean.csv`

**Cột giữ lại:** `patientunitstayid`, `observationoffset`, `heartrate`,
`systemicmean`, `sao2`, `respiration`, `temperature`.

**Xử lý:**
- Lọc cohort sớm để tiết kiệm RAM.
- **Lọc cửa sổ 24h đầu:** `0 <= observationoffset <= 1440` phút.
- **Data cleaning:**
  - `systemicmean < 20 mmHg` → `NaN` (lỗi thiết bị).
  - `temperature > 45` → coi là °F, chuyển sang °C: `(T-32)*5/9`.
  - Sau chuyển đổi, `temperature` ngoài `[30, 45]` °C → `NaN`.
  - `respiration < 5` lần/phút → `NaN`.
  - `heartrate <= 0` hoặc `> 300` → `NaN`.
  - `sao2` ngoài `(0, 100]` → `NaN`.
- **Groupby `patientunitstayid` → 9 đặc trưng:**
  - `heartrate_mean`, `heartrate_min`, `heartrate_max`
  - `systemicmean_max`
  - `sao2_mean`, `sao2_min`
  - `respiration_mean`, `respiration_max`
  - `temperature_mean`

**Kết quả:** 2.240 dòng × 10 cột.
Coverage: 96.3% cohort có dữ liệu vital.
Lưu ý: `temperature_mean` chỉ có 6.5% cohort (thưa), `systemicmean_max` 17.2%.

---

### 3.4. `lab.csv` → `lab_clean.csv`

**Cột giữ lại:** `patientunitstayid`, `labresultoffset`, `labname`, `labresult`.

**Xử lý:**
- **Lọc cửa sổ 24h đầu:** `0 <= labresultoffset <= 1440` phút.
- Ép `labresult` về số, loại `NaN`.
- **Ánh xạ `labname` về 8 nhóm chuẩn** (sau khi lowercase + strip):
  - `creatinine`
  - `bun`
  - `lactate`
  - `wbc` (`wbc`, `wbc x 1000`)
  - `hemoglobin` (`hemoglobin`, `hgb`)
  - `platelets` (`platelets`, `platelets x 1000`)
  - `bilirubin` (`bilirubin`, `total bilirubin`)
  - `glucose` (`glucose`, `bedside glucose`)
- **Groupby `patientunitstayid` → 10 đặc trưng:**
  - `creatinine_max`
  - `bun_mean`
  - `lactate_mean`, `lactate_max`
  - `wbc_mean`
  - `hemoglobin_mean`
  - `platelets_mean`
  - `bilirubin_mean`
  - `glucose_mean`, `glucose_max`
- **Outlier cực đoan:** `wbc_mean > 100 K/µL` → `NaN` (lỗi nhập liệu).

**Kết quả:** 2.265 dòng × 11 cột.
Coverage: 92.9% cohort có xét nghiệm.
Lưu ý: `lactate` chỉ có ~21% cohort (đặc điểm lâm sàng — chỉ đo khi nghi nhiễm trùng huyết).

---

## 4. Hợp nhất 4 bảng → `ai_ready.csv`

**Phương pháp:** Left join từ `patient_clean` để đảm bảo **đủ 2.326 bệnh nhân**.
Bệnh nhân thiếu bảng nào → cột đó `NaN` (XGBoost tự xử lý).

**Cấu trúc `ai_ready.csv` (25 cột):**

| Nhóm | Cột | Số lượng |
|------|-----|----------|
| ID | `patientunitstayid` | 1 |
| Nhân khẩu | `age`, `male_sex`, `bmi` | 3 |
| Thần kinh | `gcs` | 1 |
| Sinh hiệu | `heartrate_mean/min/max`, `systemicmean_max`, `sao2_mean/min`, `respiration_mean/max`, `temperature_mean` | 9 |
| Xét nghiệm | `creatinine_max`, `bun_mean`, `lactate_mean/max`, `wbc_mean`, `hemoglobin_mean`, `platelets_mean`, `bilirubin_mean`, `glucose_mean/max` | 10 |
| **Nhãn** | `mortality` | 1 |

**Kích thước cuối:** 2.326 dòng × 25 cột.

**Mất cân bằng nhãn:**
- Alive (0): 2.135 (91.79%)
- Expired (1): 191 (8.21%)
- `scale_pos_weight` gợi ý khi train XGBoost: **≈ 11.18**

---

## 5. Kiểm soát chất lượng dữ liệu

Module `data_quality.py` cung cấp các hàm kiểm tra tái sử dụng:

- `check_overview`: shape, kiểu dữ liệu, missing, trùng lặp.
- `check_plausibility`: ngưỡng sinh lý cho từng cột.
- `check_distribution`: phân bố, skewness, mất cân bằng nhãn.
- `check_hidden_issues`: sentinel, cột hằng số, cột toàn NaN.
- `check_logic`: ràng buộc logic giữa các cột.
- `audit`: chạy toàn bộ kiểm tra.

Bốn notebook kiểm tra tương ứng: `patient_quality.ipynb`,
`gcs_quality.ipynb`, `vital_quality.ipynb`, `lab_quality.ipynb`.

---

## 6. Cấu trúc thư mục

```text
ICU_Mortality_Prediction/
├── data/
│   ├── raw/eicu/          # 4 file CSV gốc
│   ├── processed/         # 4 file *_clean.csv + ai_ready.csv
│   ├── quality_checks/    # data_quality.py + 4 notebook quality
│   └── explore_data/      # notebook khám phá ban đầu
├── data_processing/       # 4 notebook process + merge_features.ipynb
├── src/                   # train.py, evaluate.py
├── document/              # tài liệu, bài báo tham khảo
└── outputs/               # kết quả mô hình
---
```

## 7. Ghi chú kỹ thuật

- **Giá trị `NaN` được giữ nguyên** thay vì điền (impute), vì XGBoost
  tự học hướng đi cho missing — và bản thân việc thiếu dữ liệu mang tín hiệu
  lâm sàng (ví dụ: nhóm có `temperature` có tỷ lệ tử vong cao gấp 2.2 lần
  nhóm thiếu).
- **Plausibility filter không xóa dòng**, chỉ gán `NaN` — giữ tối đa thông tin.
- **Không merge với cohort ở bước xử lý từng bảng** — mỗi bảng lưu độc lập,
  merge cuối cùng một lần trong `merge_features.ipynb`.
---