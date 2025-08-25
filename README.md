# PhotoRate (Flask)

사진을 업로드하면 자동으로 평점을 계산(1.0~5.0★)해서 보여주는 간단한 Flask 앱입니다.
휴리스틱(밝기/대비/선명도/색감/간단한 구도)을 조합해 점수를 냅니다.

## 빠른 시작

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# (선택) .env 설정
cp .env.example .env

# DB 초기화
flask --app app.py init-db

# 실행
python app.py
# 또는
# flask --app app.py run --debug
```

브라우저에서 http://127.0.0.1:5000 접속 → 업로드 후 평점 확인

## 구조
```
.
├─ app.py
├─ requirements.txt
├─ .env.example   # .env로 복사해서 환경설정
├─ uploads/       # 업로드 파일
├─ templates/
│  ├─ base.html
│  ├─ index.html
│  ├─ gallery.html
│  └─ detail.html
└─ static/
   └─ css/style.css
```

## 점수 산정(개요)
- **노출(exposure)**: 평균 휘도값이 과다/저노출이면 감점
- **대비(contrast)**: 표준편차로 근사
- **선명도(sharpness)**: 라플라시안 분산
- **색감(colorfulness)**: Hasler & Süsstrunk 지표
- **구도(rule-of-thirds)**: 에지 중심점이 삼등분선 근처일수록 가점

> 이 평점은 *미학적 완성도*를 정밀하게 보장하지 않습니다. 재미/데모용 휴리스틱입니다.
> 실제 미감 평가는 NIMA/LAION aesthetic 등 ML모델을 붙이는 것을 권장합니다.

## 라이선스
MIT
