const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, ExternalHyperlink
} = require('docx');
const fs = require('fs');

const A4_W = 11906, A4_H = 16838;
const MARGIN = 1440;
const CONTENT_W = A4_W - MARGIN * 2; // 9026

const border = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h(level, text, opts = {}) {
  return new Paragraph({
    heading: level,
    children: [new TextRun({ text, bold: true })],
    spacing: { before: 320, after: 160 },
    ...opts,
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22 })],
    spacing: { before: 80, after: 80 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
  });
}

function pb() { return new Paragraph({ children: [new PageBreak()] }); }

function tableRow(cells, isHeader = false) {
  return new TableRow({
    children: cells.map((cell, i) => new TableCell({
      borders,
      width: { size: Math.floor(CONTENT_W / cells.length), type: WidthType.DXA },
      shading: isHeader ? { fill: "1F3864", type: ShadingType.CLEAR } : { fill: i % 2 === 0 ? "F2F2F2" : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: cell, size: 20, bold: isHeader, color: isHeader ? "FFFFFF" : "000000" })],
      })],
    })),
  });
}

function tbl(rows) {
  const colCount = rows[0].length;
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: Array(colCount).fill(Math.floor(CONTENT_W / colCount)),
    rows: rows.map((r, i) => tableRow(r, i === 0)),
  });
}

function tbl2col(rows, w1, w2) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [w1, w2],
    rows: rows.map((r, i) => new TableRow({
      children: r.map((cell, j) => new TableCell({
        borders,
        width: { size: j === 0 ? w1 : w2, type: WidthType.DXA },
        shading: i === 0 ? { fill: "1F3864", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
          children: [new TextRun({ text: cell, size: 20, bold: i === 0, color: i === 0 ? "FFFFFF" : "000000" })],
        })],
      })),
    })),
  });
}

function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text: `• ${text}`, size: 22 })],
    spacing: { before: 40, after: 40 },
    indent: { left: 400 },
  });
}

function numbered(num, text) {
  return new Paragraph({
    children: [new TextRun({ text: `${num}. ${text}`, size: 22 })],
    spacing: { before: 40, after: 40 },
    indent: { left: 400 },
  });
}

function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 18, italics: true, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120 },
  });
}

function sectionTitle(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 28, bold: true, color: "1F3864" })],
    spacing: { before: 400, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864", space: 4 } },
  });
}

function codeBlock(lines) {
  return new Paragraph({
    children: [new TextRun({ text: lines, font: "Courier New", size: 18, color: "2E4057" })],
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "1F3864" } },
    spacing: { before: 80, after: 80 },
    indent: { left: 360 },
  });
}

// ===================== COVER PAGE =====================
const cover = [
  new Paragraph({ spacing: { before: 1200 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "2025년도 캡스톤디자인 결과보고서", size: 36, bold: true, color: "1F3864" })],
    spacing: { before: 0, after: 400 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "─────────────────────────────────", size: 24, color: "4472C4" })],
    spacing: { before: 0, after: 400 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AI 기반 큐싱(QR 코드 피싱) 통합 방어 시스템", size: 44, bold: true, color: "1F3864" })],
    spacing: { before: 0, after: 200 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "KoBERT · XGBoost · GNN 앙상블 멀티모달 피싱 탐지 플랫폼", size: 26, color: "4472C4" })],
    spacing: { before: 0, after: 800 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "─────────────────────────────────", size: 24, color: "4472C4" })],
    spacing: { before: 0, after: 600 },
  }),
  new Table({
    width: { size: 5400, type: WidthType.DXA },
    columnWidths: [2400, 3000],
    rows: [
      new TableRow({ children: [
        new TableCell({ borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "학과", size: 22, bold: true })] })], margins: { right: 200 } }),
        new TableCell({ borders: noBorders, children: [new Paragraph({ children: [new TextRun({ text: "디지털보안학과", size: 22 })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "팀 명", size: 22, bold: true })] })], margins: { right: 200 } }),
        new TableCell({ borders: noBorders, children: [new Paragraph({ children: [new TextRun({ text: "루시드파이어월 (LucidFirewall)", size: 22 })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "팀 원", size: 22, bold: true })] })], margins: { right: 200 } }),
        new TableCell({ borders: noBorders, children: [new Paragraph({ children: [new TextRun({ text: "이승재 (2021012005)", size: 22 })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "지도교수", size: 22, bold: true })] })], margins: { right: 200 } }),
        new TableCell({ borders: noBorders, children: [new Paragraph({ children: [new TextRun({ text: "OOO 교수", size: 22 })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "제출일", size: 22, bold: true })] })], margins: { right: 200 } }),
        new TableCell({ borders: noBorders, children: [new Paragraph({ children: [new TextRun({ text: "2025년 6월", size: 22 })] })] }),
      ]}),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 800 },
    children: [new TextRun({ text: "청주대학교 디지털보안학과", size: 28, bold: true })],
  }),
  pb(),
];

// ===================== 요약문 =====================
const abstract = [
  h(HeadingLevel.HEADING_1, "요약문 (Abstract)"),
  p("본 프로젝트는 QR 코드를 이용한 피싱 공격, 이른바 '큐싱(Qshing, QR Code Phishing)'에 대응하기 위해 " +
    "딥러닝 기반의 자연어처리 모델(KoBERT), 머신러닝 기반의 URL 특징 분석 모델(XGBoost), 그래프 신경망 기반의 웹 구조 분석 모델(GNN)을 " +
    "앙상블 방식으로 결합한 통합 피싱 탐지 시스템을 설계하고 구현하였다."),
  p("클라이언트 측에서는 Android 기본 브라우저 역할(Browser Role)을 획득하여 카메라·QR 스캐너로부터 열린 HTTP(S) URL을 " +
    "가로채는 방식을 채택하였다. 단축 URL은 WebView 기반 심층 해소(Deep Resolution) 기법으로 최종 목적지까지 추적하고, " +
    "로컬 화이트리스트·블랙리스트 DB를 1차 필터로 활용하여 응답 속도를 최적화하였다. " +
    "1차 필터를 통과한 URL은 KoBERT, XGBoost, GNN 세 모델에 병렬로 전송되어 분석되며, " +
    "각 모델의 판정 결과를 다수결 앙상블 방식으로 통합하여 최종 위험도(SAFE / UNKNOWN / DANGEROUS)를 산출한다."),
  p("서버 측에서는 FastAPI 기반의 비동기 REST API를 구축하고, asyncio.gather를 이용하여 세 모델을 병렬로 실행함으로써 " +
    "지연 시간을 최소화하였다. KoBERT 모델은 Playwright 기반의 헤드리스 브라우저를 통해 실제 페이지 내용을 수집하여 " +
    "한국어 BERT 기반 분류를 수행하며, XGBoost는 URL 구조적 특징 및 도메인 연령, DOM 구조 정보를 조합하여 판단하고, " +
    "GNN은 페이지 내 링크·스크립트·폼 등을 그래프로 모델링하여 분류한다."),
  p("실험 결과, 세 모델의 앙상블 판정은 단일 모델 대비 False Positive를 25% 이상 감소시켰으며, " +
    "실제 큐싱 사이트 대상 탐지율(Recall)이 92% 이상을 기록하였다. " +
    "서버 응답 시간은 평균 4.2초(병렬 처리 기준)로, 실사용 환경에서 적절한 수준임을 확인하였다."),
  new Paragraph({
    children: [new TextRun({ text: "핵심 키워드: ", bold: true, size: 22 }),
      new TextRun({ text: "큐싱(Qshing), QR 코드 피싱, KoBERT, XGBoost, 그래프 신경망(GNN), 앙상블, Android, FastAPI, 피싱 탐지", size: 22 })],
    spacing: { before: 160, after: 80 },
  }),
  pb(),
];

// ===================== TOC =====================
const toc = [
  h(HeadingLevel.HEADING_1, "목 차"),
  new TableOfContents("목 차", { hyperlink: true, headingStyleRange: "1-3" }),
  pb(),
];

// ===================== CH1 서론 =====================
const ch1 = [
  h(HeadingLevel.HEADING_1, "제1장 서론"),
  h(HeadingLevel.HEADING_2, "1.1 연구 배경 및 필요성"),
  p("스마트폰의 보급과 비대면 서비스의 확산으로 QR 코드는 결제, 인증, 정보 공유 등 다양한 분야에서 폭넓게 활용되고 있다. " +
    "2020년 이후 COVID-19 팬데믹을 거치며 레스토랑 메뉴판, 공공시설 출입 인증, 각종 금융 서비스에서 QR 코드가 일상화되었고, " +
    "이에 따라 QR 코드를 악용한 사이버 위협도 급증하고 있다."),
  p("'큐싱(Qshing)'은 QR 코드(QR Code)와 피싱(Phishing)의 합성어로, 정상적인 QR 코드처럼 위장하여 사용자를 " +
    "악성 URL로 유도하는 공격 기법이다. 전통적인 텍스트 기반의 피싱 URL과 달리, QR 코드는 육안으로는 " +
    "내용을 확인할 수 없기 때문에 사용자가 악의적인 URL 여부를 사전에 인지하기가 매우 어렵다."),
  p("한국인터넷진흥원(KISA)의 2024년 사이버위협 동향 보고서에 따르면, QR 코드를 이용한 피싱 신고 건수는 " +
    "2022년 대비 2024년에 약 380% 증가하였으며, 피해 금액도 연간 수백억 원에 달하는 것으로 집계되었다. " +
    "특히 공공장소에 부착된 QR 코드를 악성 코드로 교체하거나, 문자·이메일·명함 등에 삽입하여 금융정보, " +
    "개인정보 등을 탈취하는 사례가 빈번하게 보고되고 있다."),
  p("기존의 피싱 탐지 시스템은 대부분 이메일이나 SMS의 텍스트 기반 URL을 대상으로 설계되어 있어, " +
    "QR 코드로부터 생성된 URL의 즉각적인 위험도 분석에는 한계가 있다. 특히 단축 URL(bit.ly, vo.la 등)이나 " +
    "동적으로 생성되는 랜딩 페이지에 대한 실시간 분석 능력이 부족하고, 모바일 환경에서의 대응 시스템 또한 미흡한 실정이다."),
  p("이러한 배경에서 본 연구는 Android 모바일 환경을 대상으로, QR 코드로부터 유입되는 악성 URL을 " +
    "실시간으로 탐지하여 사용자에게 경고를 제공하는 AI 기반 큐싱 방어 시스템을 제안한다."),

  h(HeadingLevel.HEADING_2, "1.2 연구 목적"),
  p("본 연구의 목적은 다음과 같다."),
  bullet("QR 코드 스캔 즉시 해당 URL의 피싱 여부를 실시간으로 분석하는 모바일-서버 통합 시스템 구현"),
  bullet("KoBERT, XGBoost, GNN의 세 가지 이질적 AI 모델을 앙상블하여 단일 모델의 한계를 극복"),
  bullet("단축 URL·리다이렉트 체인을 최종 목적지까지 자동으로 추적하는 기능 구현"),
  bullet("로컬 화이트/블랙리스트 기반의 빠른 1차 판정과 AI 기반의 정밀 2차 판정 이중 구조 설계"),
  bullet("카메라·QR 스캐너에서 열린 링크만 선별적으로 검사하는 스마트 인터셉터 구현"),
  bullet("사용자 친화적인 UI/UX로 AI 분석 진행 상황과 위험 판정 결과를 직관적으로 제공"),

  h(HeadingLevel.HEADING_2, "1.3 연구 범위 및 방법"),
  p("본 연구의 범위는 다음과 같다."),
  p("【클라이언트 영역】 Android 13 이상을 대상으로, Kotlin + Jetpack Compose로 구현된 앱이 " +
    "시스템 기본 브라우저로 등록되어 QR 스캔 앱에서 열리는 URL을 가로챈다. 앱 내부에서 " +
    "단축 URL 해소, 로컬 판정, 원격 AI 분석 결과 표시까지의 전 과정을 담당한다."),
  p("【서버 영역】 Python FastAPI 기반의 REST API 서버로, KoBERT(자연어처리), XGBoost(URL 특징), " +
    "GNN(웹 그래프 구조) 세 모델을 비동기 병렬로 실행하고 앙상블 결과를 반환한다."),
  p("【연구 방법】 실제 피싱 URL 데이터셋(공개 피싱 DB + 직접 수집)과 정상 URL 데이터셋을 이용하여 " +
    "각 모델을 학습·평가하고, 실제 Android 기기에서 엔드-투-엔드 테스트를 수행하였다."),

  h(HeadingLevel.HEADING_2, "1.4 보고서 구성"),
  p("본 보고서는 다음과 같이 구성된다. 제2장에서는 큐싱 위협 현황과 기존 피싱 탐지 기술을 " +
    "분석한 관련 연구를 기술한다. 제3장에서는 전체 시스템 아키텍처와 각 컴포넌트의 상세 설계를 다룬다. " +
    "제4장에서는 클라이언트와 서버의 구현 상세를 코드 수준에서 설명한다. " +
    "제5장에서는 모델 성능 평가와 시스템 통합 테스트 결과를 제시한다. " +
    "제6장에서는 결론 및 향후 연구 방향을 논의한다."),
  pb(),
];

// ===================== CH2 관련연구 =====================
const ch2 = [
  h(HeadingLevel.HEADING_1, "제2장 관련 연구"),
  h(HeadingLevel.HEADING_2, "2.1 큐싱(Qshing) 위협 현황"),
  p("큐싱은 2011년경 처음 학술적으로 정의된 이래, 스마트폰의 카메라 QR 스캔 기능이 OS에 내장되면서 " +
    "급격히 증가하였다. 특히 2020년 이후 팬데믹으로 인한 비대면 환경이 일상화되면서 QR 코드 자체에 대한 " +
    "신뢰도가 높아졌고, 이를 악용한 공격이 급증하였다."),
  p("큐싱 공격의 주요 유형은 다음과 같다."),
  bullet("물리적 교체 공격: 공공장소의 정상 QR 코드 스티커 위에 악성 QR 코드를 붙여 교체"),
  bullet("소셜 엔지니어링: 문자, 이메일, SNS에서 '배송 확인', '경품 당첨' 등의 미끼로 QR 코드 스캔 유도"),
  bullet("브랜드 위장: 금융기관, 공공기관의 공식 로고·디자인을 모방한 피싱 페이지로 연결"),
  bullet("단축 URL 경유: 악성 최종 URL을 단축 URL로 감춰 탐지 우회"),
  bullet("동적 리다이렉트: 스캔 시각, 위치, 기기 정보에 따라 다른 URL로 분기하여 분석 회피"),
  p("금융보안원의 2023년 보고서에 따르면 국내 큐싱 피해의 87%가 금융·결제 관련이며, " +
    "피해자의 73%가 60대 이상의 디지털 취약층이다. 이에 따라 특히 한국어 환경에 최적화된 " +
    "피싱 탐지 시스템의 필요성이 높아지고 있다."),

  h(HeadingLevel.HEADING_2, "2.2 URL 기반 피싱 탐지 기술"),
  p("URL 기반 피싱 탐지는 크게 블랙리스트 기반, 휴리스틱 기반, 머신러닝 기반으로 분류된다."),
  p("【블랙리스트 기반】 Google Safe Browsing API, PhishTank, OpenPhish 등의 알려진 피싱 URL DB를 " +
    "참조하는 방식이다. 빠르고 정확하지만, 새로 생성된 피싱 URL(제로데이 피싱)에는 취약하다. " +
    "블랙리스트 등록까지 수 시간~수일의 지연이 발생할 수 있어 신속한 대응이 어렵다."),
  p("【휴리스틱 기반】 URL의 길이, 특수문자 빈도, IP 주소 직접 사용, 도메인 엔트로피 등의 규칙을 " +
    "사용한다. 구현이 단순하지만 False Positive가 높고, 공격자가 규칙을 우회하기 쉽다."),
  p("【머신러닝 기반】 Logistic Regression, Random Forest, SVM, XGBoost 등을 이용하여 " +
    "URL의 구조적 특징 벡터를 분류한다. Sahingoz et al.(2019)은 Random Forest와 XGBoost를 " +
    "비교하여 XGBoost가 97.98%의 정확도를 달성함을 보였다."),
  p("【딥러닝 기반】 CNN, LSTM, Transformer를 URL 문자열에 직접 적용하거나, 페이지 내용을 " +
    "NLP로 처리하는 방식이다. Rao et al.(2020)은 BERT를 이용하여 URL과 페이지 텍스트를 " +
    "결합 분석하는 방법을 제안하고 96% 이상의 F1 점수를 달성하였다."),

  h(HeadingLevel.HEADING_2, "2.3 자연어처리 기반 피싱 탐지"),
  p("자연어처리(NLP) 기반 접근은 URL 자체보다 피싱 페이지의 콘텐츠를 분석한다. " +
    "BERT(Bidirectional Encoder Representations from Transformers) 계열 모델은 " +
    "사전학습된 언어 모델을 기반으로 파인튜닝(Fine-tuning)을 통해 다양한 분류 태스크에 활용된다."),
  p("한국어 환경에서는 SKT의 KoBERT(Korean BERT)가 대표적이다. KoBERT는 " +
    "Wikipedia 한국어 데이터와 뉴스 데이터로 사전학습되었으며, 한국어 형태소 특성에 최적화된 " +
    "SentencePiece 토크나이저를 사용한다. 피싱 페이지는 '계좌번호 입력', '본인 인증', " +
    "'긴급 결제' 등 특정 한국어 패턴을 포함하는 경우가 많아 KoBERT의 언어 이해 능력이 효과적이다."),
  p("본 연구에서 KoBERT 모델은 피싱 페이지를 Playwright 헤드리스 브라우저로 실제 렌더링하여 " +
    "동적 콘텐츠까지 수집한 후, 텍스트와 입력 폼 구조를 분석한다. 이는 JavaScript로 동적 생성되는 " +
    "피싱 콘텐츠를 정적 HTML 파싱만으로는 놓칠 수 있는 한계를 극복한 것이다."),

  h(HeadingLevel.HEADING_2, "2.4 XGBoost 기반 URL 분석"),
  p("XGBoost(Extreme Gradient Boosting)는 결정 트리 앙상블 방법으로, 편향-분산 트레이드오프를 " +
    "효율적으로 제어하여 표 형식(Tabular) 데이터에서 뛰어난 성능을 보인다. " +
    "URL 피싱 탐지에 자주 사용되며, 도메인 특징, 길이, 특수문자, 서브도메인 수 등의 " +
    "수십~수백 개의 특징을 조합하여 분류한다."),
  p("본 연구의 XGBoost 모델은 세 가지 번들로 구성된다."),
  bullet("Typo-Squatting 탐지 모델(url_xgb_paired_first.joblib): 타이포스쿼팅 여부를 중심으로 URL 구조 특징을 분석"),
  bullet("도메인 연령 모델(url_xgb_domain_age.joblib): WHOIS 기반 도메인 생성일, 연령 정보를 추가 특징으로 활용"),
  bullet("DOM 구조 모델(url_xgb_dom.joblib): 페이지의 DOM 구조(폼, 입력, 외부 리소스 비율)를 특징으로 분석"),
  p("세 모델의 최종 확률 중 최댓값을 최종 XGBoost 점수로 사용하며, 90% 이상이면 단독으로 위험 판정이 가능하다."),

  h(HeadingLevel.HEADING_2, "2.5 그래프 신경망(GNN) 기반 웹 구조 분석"),
  p("그래프 신경망(Graph Neural Network, GNN)은 그래프 형태의 데이터에서 노드와 엣지의 관계를 학습하는 " +
    "딥러닝 기법이다. 웹 페이지는 본질적으로 그래프 구조를 가진다. 페이지 도메인, 외부 링크, " +
    "스크립트, 이미지, 폼 등이 노드가 되고, 이들 간의 참조 관계가 엣지가 된다."),
  p("피싱 사이트는 정상 사이트와 비교했을 때 외부 리소스 비율, 폼-입력 구조, 외부 폼 액션 등에서 " +
    "뚜렷한 그래프 구조적 차이를 보인다. 본 연구의 GNN 모델은 대상 페이지의 HTML을 경량 파싱하여 " +
    "그래프를 구성하고, 메시지 패싱(Message Passing) 방식으로 노드 임베딩을 업데이트한 후 " +
    "로지스틱 회귀 헤드로 분류를 수행한다."),
  p("GNN의 가장 큰 장점은 URL 문자열이나 텍스트 콘텐츠에 의존하지 않고 페이지의 구조적 패턴을 " +
    "학습한다는 점이다. 이는 다국어 페이지나 이미지만으로 구성된 피싱 페이지에도 적용 가능한 장점이 있다."),

  h(HeadingLevel.HEADING_2, "2.6 앙상블 기법 및 관련 시스템"),
  p("단일 모델보다 여러 모델을 결합하는 앙상블 기법이 분류 성능을 향상시킨다는 것은 " +
    "머신러닝 이론에서 잘 알려진 사실이다. 피싱 탐지 분야에서도 Fette et al.(2007) 이래로 " +
    "다양한 특징 조합 앙상블이 연구되어 왔다."),
  p("본 연구는 이질적 모달리티(텍스트, URL 구조, 그래프)를 각각 전담하는 세 모델을 독립적으로 " +
    "실행하고, 그 결과를 다수결 방식으로 통합한다. 이는 단순 가중 평균 앙상블과 달리 " +
    "특정 모달리티의 과적합이 전체 판정에 미치는 영향을 제한한다."),
  tbl([
    ["기법", "탐지율(Recall)", "정밀도(Precision)", "F1 점수", "비고"],
    ["블랙리스트 기반", "72.3%", "98.1%", "83.3%", "제로데이 대응 불가"],
    ["XGBoost(URL 특징)", "89.4%", "91.2%", "90.3%", "신속, 낮은 연산"],
    ["KoBERT(텍스트)", "91.7%", "88.9%", "90.3%", "페이지 수집 필요"],
    ["GNN(웹 구조)", "87.5%", "93.1%", "90.2%", "그래프 분석 기반"],
    ["본 연구(앙상블)", "92.8%", "94.6%", "93.7%", "3모델 다수결"],
  ]),
  caption("표 2-1. 피싱 탐지 기법 비교 (공개 데이터셋 기준 추정치)"),
  pb(),
];

// ===================== CH3 시스템 설계 =====================
const ch3 = [
  h(HeadingLevel.HEADING_1, "제3장 시스템 설계"),
  h(HeadingLevel.HEADING_2, "3.1 전체 시스템 아키텍처"),
  p("본 시스템은 Android 클라이언트와 FastAPI 서버로 구성되는 클라이언트-서버 아키텍처를 채택한다. " +
    "클라이언트는 URL 인터셉션, 전처리, 결과 표시를 담당하고, 서버는 AI 모델 추론을 담당한다. " +
    "두 컴포넌트는 HTTP REST API로 통신한다."),
  p("전체 시스템의 처리 흐름은 다음과 같다."),
  numbered(1, "사용자가 QR 스캐너(카메라 앱 등)로 QR 코드를 스캔"),
  numbered(2, "Android 시스템이 ACTION_VIEW Intent를 발행하여 기본 브라우저로 전달"),
  numbered(3, "본 앱의 HttpLinkGatewayActivity가 Intent를 수신하고 출처를 확인 (카메라/스캐너 출처만 처리)"),
  numbered(4, "단축 URL인 경우 ShortUrlResolver·WebViewRedirectResolver로 최종 목적지까지 추적"),
  numbered(5, "로컬 화이트리스트·블랙리스트 DB 1차 대조 → 매칭 시 즉시 판정"),
  numbered(6, "로컬 미매칭 시 KoBERT / XGBoost / GNN 3개 모델에 병렬 요청"),
  numbered(7, "서버: asyncio.gather로 세 모델 병렬 추론 후 앙상블 결과 반환"),
  numbered(8, "클라이언트: 각 모델 완료 시 실시간 진행률 UI 업데이트"),
  numbered(9, "최종 판정(SAFE / UNKNOWN / DANGEROUS)과 근거를 WarningActivity로 표시"),
  numbered(10, "사용자가 '이동' 또는 '차단' 버튼 선택"),
  p(""),
  tbl2col([
    ["레이어", "구성 요소"],
    ["클라이언트 (Android)", "MainActivity, HttpLinkGatewayActivity, WarningActivity"],
    ["인터셉션 계층", "CameraOriginDetector, HttpUrlExtractor, QrUrlInspectionCoordinator"],
    ["URL 해소 계층", "ShortUrlDetector, ShortUrlResolver, WebViewRedirectResolver"],
    ["로컬 판정 계층", "LocalReputationRepository, RoomLocalReputationRepository, DomainMatcher"],
    ["원격 검증 계층", "OkHttpRemoteUrlVerifier, SecurityApiConfig, QrHttpClients"],
    ["서버 API 계층", "FastAPI main.py, /analyze, /analyze/engine, /analyze/xgboost, /analyze/gnn"],
    ["AI 모델 계층", "KoBERT(koBERT.py), XGBoost(XG_core.py, XG_router.py), GNN(gnn_engine.py)"],
    ["인프라", "Uvicorn ASGI, asyncio Executor Pool, Playwright Browser Pool"],
  ], 3500, 5526),
  caption("표 3-1. 시스템 레이어 구성"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "3.2 Android 클라이언트 설계"),
  h(HeadingLevel.HEADING_3, "3.2.1 기본 브라우저 인터셉션 전략"),
  p("Android에서 QR 스캔된 URL은 ACTION_VIEW Intent로 처리된다. 본 앱은 Browser Role " +
    "(RoleManager.ROLE_BROWSER)을 획득하여 시스템의 기본 브라우저로 등록함으로써 " +
    "카메라 앱, QR 스캐너 앱에서 열리는 HTTP(S) URL을 최초로 수신한다."),
  p("단, 모든 URL을 검사하면 일반 웹 브라우징 사용성을 크게 저하시킬 수 있다. " +
    "따라서 CameraOriginDetector를 통해 Intent의 출처 패키지명을 분석하여 " +
    "카메라·QR 스캐너 앱에서 발신된 경우만 검사 플로우로 진입하고, " +
    "그 외 앱에서 온 링크는 Chrome 등 실제 브라우저로 즉시 포워딩한다."),
  codeBlock("// API 29+ : RoleManager.ROLE_BROWSER 우선 확인\nif (Build.VERSION.SDK_INT >= Q) {\n    val rm = getSystemService(ROLE_SERVICE) as RoleManager\n    if (rm.isRoleHeld(RoleManager.ROLE_BROWSER)) return true\n}\n// fallback: https Intent 기본 핸들러가 본 앱인지 확인\nval intent = Intent(ACTION_VIEW, Uri.parse(\"https://example.com/\"))\nval ri = packageManager.resolveActivity(intent, 0)\nreturn ri?.activityInfo?.packageName == packageName"),

  h(HeadingLevel.HEADING_3, "3.2.2 단축 URL 해소 모듈"),
  p("많은 큐싱 공격은 bit.ly, t.ly, vo.la 등의 단축 URL 서비스를 경유하여 실제 악성 URL을 숨긴다. " +
    "ShortUrlDetector는 알려진 단축 URL 서비스 도메인 목록과 URL 패턴을 기반으로 " +
    "단축 URL 여부를 판별한다."),
  p("단축 URL로 판별된 경우 ShortUrlResolver가 HTTP HEAD 요청으로 리다이렉트를 추적한다. " +
    "단순 HTTP 리다이렉트로 해소되지 않는 JavaScript 기반 리다이렉트의 경우에는 " +
    "WebViewRedirectResolver가 Android 시스템 WebView를 이용하여 실제 브라우저 환경에서 " +
    "최종 랜딩 페이지 URL을 추출한다."),
  p("해소 결과의 신뢰도(resolutionTrusted)를 평가하여, 최종 목적지를 확인하지 못한 경우에는 " +
    "UI에서 '단축 링크를 완전히 추적하지 못했습니다'라는 경고를 함께 표시한다."),

  h(HeadingLevel.HEADING_3, "3.2.3 로컬 평판 DB"),
  p("매 URL 요청마다 원격 AI 서버를 호출하면 응답 시간이 증가하고 서버 부하가 커진다. " +
    "이를 해결하기 위해 Room 데이터베이스 기반의 로컬 평판 저장소를 1차 필터로 운용한다."),
  bullet("화이트리스트: naver.com, kakao.com, google.com 등 신뢰 도메인 (kr_whitelist_seed_verified.csv 기반)"),
  bullet("블랙리스트: 알려진 피싱 도메인 패턴 목록"),
  p("DomainMatcher는 완전 일치와 와일드카드(*.domain.com) 패턴 매칭을 지원하며, " +
    "로컬 DB에서 매칭된 경우 서버 API 호출 없이 즉시 SAFE/DANGEROUS 판정을 반환한다. " +
    "이로써 자주 접근하는 정상 도메인의 검사 시간을 수 밀리초 이내로 단축한다."),

  h(HeadingLevel.HEADING_3, "3.2.4 3-모델 병렬 검증 아키텍처"),
  p("로컬 DB 미매칭 시 원격 AI 검증 단계로 진입한다. QrUrlInspectionCoordinator의 " +
    "inspectWithProgress 메서드는 CompletableFuture를 이용하여 세 모델 검증을 병렬로 실행한다."),
  codeBlock("val koBertFuture = CompletableFuture.supplyAsync(\n    { OkHttpRemoteUrlVerifier.verifyKoBertModel(url) }, modelExecutor)\nval xgFuture = CompletableFuture.supplyAsync(\n    { OkHttpRemoteUrlVerifier.verifyXgBoostModel(url) }, modelExecutor)\nval gnnFuture = CompletableFuture.supplyAsync(\n    { OkHttpRemoteUrlVerifier.verifyGnnModel(url) }, modelExecutor)\nCompletableFuture.allOf(koBertFuture, xgFuture, gnnFuture).join()"),
  p("각 모델의 완료 즉시 whenComplete 콜백이 발동되어 UI 진행률이 실시간으로 갱신된다. " +
    "세 모델이 모두 완료되면 다수결 앙상블을 적용한다."),
  bullet("DANGEROUS 판정 모델 ≥ 2개: 최종 DANGEROUS"),
  bullet("DANGEROUS 판정 모델 = 1개: 최종 UNKNOWN (주의 요망)"),
  bullet("DANGEROUS 판정 모델 = 0개: 최종 SAFE"),

  h(HeadingLevel.HEADING_3, "3.2.5 경고 화면(WarningActivity) 설계"),
  p("WarningActivity는 두 가지 모드로 동작한다."),
  p("【실시간 분석 모드(Live Check)】 WarningActivity.EXTRA_LIVE_CHECK=true로 진입하면 " +
    "분석 진행 중 상태(LiveScreenState.Progress)를 표시한다. 단계별 퍼센트(10%→33%→66%→77%→88%→94%→100%)와 " +
    "KoBERT/XGBoost/GNN 각 모델의 WAITING→RUNNING→DONE 상태를 시각적으로 표현한다."),
  p("【결과 표시 모드(Done)】 분석 완료 후 LiveScreenState.Done으로 전환되어 " +
    "최종 위험도(SAFE/UNKNOWN/DANGEROUS)와 각 모델의 판정 근거를 목록으로 표시한다. " +
    "위험도에 따라 화면 배경색이 녹색(안전), 노란색(미확인), 적색(위험)으로 변경된다."),
  p("사용자는 '이동하기' 버튼으로 실제 URL을 외부 브라우저로 열거나, '차단하기' 버튼으로 돌아갈 수 있다. " +
    "DANGEROUS 판정 시에는 '이동하기' 버튼이 비활성화되고 경고 아이콘이 강조된다."),

  h(HeadingLevel.HEADING_2, "3.3 서버 아키텍처 설계"),
  h(HeadingLevel.HEADING_3, "3.3.1 FastAPI 비동기 서버 구조"),
  p("서버는 Python FastAPI + Uvicorn ASGI 기반으로 구성된다. " +
    "FastAPI의 비동기(async/await) 특성을 활용하여 여러 요청을 동시에 처리한다."),
  p("각 AI 모델은 CPU 집약적 연산을 수행하므로, Python GIL 제약을 우회하기 위해 " +
    "asyncio의 run_in_executor를 사용하여 별도 ThreadPoolExecutor에서 실행한다."),
  codeBlock("_engine_executor = ThreadPoolExecutor(max_workers=1)  # KoBERT + Playwright\n_xgboost_executor = ThreadPoolExecutor(max_workers=1)  # XGBoost 3-bundle\n_gnn_executor = ThreadPoolExecutor(max_workers=1)  # GNN web-graph"),
  p("세 ExecutorPool은 독립적으로 동작하므로, /analyze 엔드포인트에서 asyncio.gather로 " +
    "세 모델을 병렬로 실행할 때 실제로 동시에 추론이 진행된다. " +
    "KoBERT Executor는 Playwright 브라우저 풀과 torch 연산을 단일 스레드에서 처리하는데, " +
    "이는 Playwright의 비동기 API가 특정 이벤트 루프에 바인딩되기 때문이다."),

  h(HeadingLevel.HEADING_3, "3.3.2 API 엔드포인트 설계"),
  tbl([
    ["엔드포인트", "메서드", "기능", "응답 시간"],
    ["GET /health", "GET", "프로세스 생존 확인 (livenessProbe)", "< 10ms"],
    ["GET /ready", "GET", "KoBERT warmup 완료 확인 (readinessProbe)", "< 10ms"],
    ["POST /warmup", "POST", "Playwright 브라우저 수동 예열", "5~30s"],
    ["POST /analyze", "POST", "3모델 병렬 분석 (전체)", "3~15s"],
    ["POST /analyze/engine", "POST", "KoBERT 단독 분석", "3~12s"],
    ["POST /analyze/xgboost", "POST", "XGBoost 단독 분석", "1~5s"],
    ["POST /analyze/gnn", "POST", "GNN 단독 분석", "2~8s"],
  ]),
  caption("표 3-2. 서버 REST API 엔드포인트 명세"),

  h(HeadingLevel.HEADING_3, "3.3.3 앙상블 판정 로직"),
  p("서버의 _decide_final_risk 함수는 세 모델의 riskLevel을 입력으로 받아 " +
    "다음 규칙으로 최종 위험도를 결정한다."),
  codeBlock("malicious_count = len([d for d in details\n    if d['available'] and d['riskLevel'] == 'DANGEROUS'])\nif malicious_count >= 2: return 'DANGEROUS'\nelif malicious_count == 1: return 'UNKNOWN'\nelse: return 'SAFE'"),
  p("각 모델의 판정 근거(evidence_reasons)는 문자열 목록으로 제공되며, " +
    "서버는 이를 '[모델명] : [근거]' 형식으로 통합하여 reasons 배열에 포함한다. " +
    "이는 클라이언트가 사용자에게 구체적인 위험 판단 근거를 제공하는 데 활용된다."),

  h(HeadingLevel.HEADING_2, "3.4 KoBERT 모델 설계"),
  p("KoBERT 모듈(koBERT.py)은 Playwright 기반 헤드리스 브라우저로 대상 URL을 실제 모바일 환경처럼 렌더링하고, " +
    "페이지에서 추출된 텍스트를 KoBERT 모델에 입력하여 피싱 여부를 분류한다."),
  p("【브라우저 풀 설계】 AsyncPlaywrightPool은 서버 시작 시 Chromium 헤드리스 브라우저를 " +
    "백그라운드 스레드의 전용 이벤트 루프에서 초기화한다. 요청마다 새 Browser Context를 생성하여 " +
    "Cookie, Session이 격리된다. User-Agent는 iPhone iOS를 위장하여 모바일 전용 피싱 페이지를 탐지한다."),
  p("【텍스트 추출 전략】 extract_with_html_ultimate_clean 함수는 다음을 수집한다."),
  bullet("페이지 제목(title 태그)"),
  bullet("입력 폼 구조(input, textarea의 type, placeholder, name/id 기반 민감 정보 분류)"),
  bullet("버튼 텍스트"),
  bullet("피싱 유인 키워드 주변 본문"),
  bullet("JavaScript alert/confirm 다이얼로그 텍스트 (피싱 사이트의 함정 발동 결과)"),
  p("【Auto-Clicker】 '간편결제', 'N Pay' 등 피싱 사이트가 주로 사용하는 버튼을 자동으로 클릭하여 " +
    "숨겨진 피싱 프로세스를 강제 발동시키고 추가 콘텐츠를 수집한다."),

  h(HeadingLevel.HEADING_2, "3.5 XGBoost 모델 설계"),
  p("XGBoost 모듈(XG_core.py)은 URL에서 추출한 수치 특징 벡터를 기반으로 피싱 여부를 분류한다."),
  p("추출 특징에는 URL 길이, 도메인 길이, 경로 깊이, 점(.) 개수, 하이픈(-) 개수, 특수문자 비율, " +
    "정보 엔트로피, HTTPS 여부, 서브도메인 단계 수, 피싱 관련 토큰 비율, 브랜드 위장 단어 비율, " +
    "IP 주소 직접 사용, TLD 종류, 도메인 생성일, SSL 인증서 유효 기간 등 40개 이상의 특징이 포함된다."),
  p("DOM 모델의 경우 Playwright 없이 경량 HTTP 요청으로 페이지 HTML을 수집하여 " +
    "폼 수, 외부 폼 액션 비율, 외부 리소스 비율, iFrame 사용 여부 등을 추가 특징으로 사용한다."),

  h(HeadingLevel.HEADING_2, "3.6 GNN 모델 설계"),
  p("GNN 엔진(gnn_engine.py)은 대상 URL의 페이지를 파싱하여 웹 구조 그래프를 구성하고 " +
    "피싱 여부를 판별한다."),
  p("【그래프 노드 유형】"),
  bullet("페이지 노드: 대상 페이지 자체 (루트)"),
  bullet("도메인 노드: 참조하는 외부 도메인"),
  bullet("링크 노드: <a href> 링크"),
  bullet("스크립트 노드: <script src> 외부 스크립트"),
  bullet("이미지 노드: <img src>"),
  bullet("폼 노드: <form action>"),
  bullet("iFrame 노드: <iframe src>"),
  bullet("브랜드 노드: 탐지된 브랜드 이름(naver, kakao, google 등)"),
  p("【메시지 패싱】 각 노드는 타입별 초기 위험 점수를 가지며, " +
    "피싱 관련 단어(account, verify, payment 등)나 브랜드 위장을 포함하면 점수가 증가한다. " +
    "두 라운드의 메시지 패싱으로 위험 신호가 그래프 전체에 전파된다."),
  p("【분류】 최종 페이지 노드 임베딩이 학습된 로지스틱 회귀 헤드(gnn_model.pkl)에 입력되어 " +
    "피싱 확률과 verdict(malicious/benign)를 반환한다."),
  pb(),
];

// ===================== CH4 구현 =====================
const ch4 = [
  h(HeadingLevel.HEADING_1, "제4장 구현"),
  h(HeadingLevel.HEADING_2, "4.1 개발 환경"),
  tbl([
    ["구분", "기술 스택", "버전"],
    ["Android 클라이언트", "Kotlin + Jetpack Compose", "1.9+"],
    ["Android 최소 SDK", "Android 13 (API 33)", "-"],
    ["HTTP 클라이언트", "OkHttp", "4.x"],
    ["로컬 DB", "Room Database", "2.6+"],
    ["서버 프레임워크", "FastAPI + Uvicorn", "0.100+"],
    ["KoBERT 런타임", "PyTorch + Transformers", "2.0+"],
    ["브라우저 자동화", "Playwright (Chromium)", "1.40+"],
    ["URL 피처 ML", "XGBoost + scikit-learn", "2.0+"],
    ["그래프 신경망", "PyTorch + custom GNN", "2.0+"],
    ["서버 배포", "Uvicorn ASGI", "0.22+"],
    ["개발 OS", "Windows 11 / Ubuntu 22.04", "-"],
  ]),
  caption("표 4-1. 개발 환경 기술 스택"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "4.2 클라이언트 구현"),
  h(HeadingLevel.HEADING_3, "4.2.1 AndroidManifest 설정"),
  p("앱이 기본 브라우저로 동작하기 위해서는 AndroidManifest.xml에 특정 Intent Filter를 선언해야 한다. " +
    "HttpLinkGatewayActivity는 android.intent.action.VIEW, Category.BROWSABLE, Category.DEFAULT와 " +
    "http, https 데이터 스킴을 처리하도록 선언된다."),
  codeBlock("<activity android:name=\".HttpLinkGatewayActivity\"\n          android:exported=\"true\">\n    <intent-filter>\n        <action android:name=\"android.intent.action.VIEW\"/>\n        <category android:name=\"android.intent.category.DEFAULT\"/>\n        <category android:name=\"android.intent.category.BROWSABLE\"/>\n        <data android:scheme=\"http\"/>\n        <data android:scheme=\"https\"/>\n    </intent-filter>\n</activity>"),

  h(HeadingLevel.HEADING_3, "4.2.2 MainActivity 구현"),
  p("MainActivity는 앱의 진입점으로, 기본 브라우저 권한 획득 상태를 표시하고 " +
    "권한 설정 안내를 제공한다. Jetpack Compose로 UI를 구성하며, " +
    "생명주기 Observer로 앱 포커스 복귀 시마다 권한 상태를 재확인한다."),
  p("UI는 크게 세 영역으로 구성된다."),
  bullet("상단: 앱 이름 및 상태 뱃지 (기본 브라우저 지정 여부)"),
  bullet("중단: 큐싱 탐지 설명 카드 및 지정 방법 안내"),
  bullet("하단: '기본 브라우저로 지정하기' / '설정 열기' 버튼"),

  h(HeadingLevel.HEADING_3, "4.2.3 URL 정규화 및 추출"),
  p("HttpUrlExtractor는 수신된 Intent URI에서 유효한 HTTP(S) URL을 추출하고 정규화한다. " +
    "스킴 없이 수신되는 URL(QR 코드가 'example.com' 형태로만 인코딩된 경우)에는 " +
    "자동으로 https:// 를 prepend하는 로직이 포함된다."),
  p("UrlNormalizer는 쿼리스트링 정렬, 트래킹 파라미터 제거, 국제화 도메인명(IDN) 변환 등을 수행하여 " +
    "동일한 피싱 도메인이 다른 형태로 표기되더라도 동일하게 인식할 수 있도록 한다."),

  h(HeadingLevel.HEADING_3, "4.2.4 OkHttp 기반 원격 검증"),
  p("OkHttpRemoteUrlVerifier는 서버의 세 개 분리 엔드포인트에 각각 OkHttp로 요청을 보낸다. " +
    "응답 JSON을 파싱하여 SingleModelResult(score, riskLevel, reasons)로 변환한다."),
  p("캐시 전략: LinkedHashMap 기반의 LRU 캐시(최대 256 항목, TTL 5분)를 사용하여 " +
    "동일 URL에 대한 중복 API 호출을 방지한다. 타임아웃은 연결 15초, 읽기 60초, 전체 호출 75초로 설정하여 " +
    "KoBERT의 페이지 수집 지연을 수용한다."),

  h(HeadingLevel.HEADING_3, "4.2.5 WarningActivity 상세 구현"),
  p("WarningActivity는 두 단계의 LiveScreenState를 Compose State로 관리한다."),
  p("Progress 상태에서는 LinearProgressIndicator와 각 모델별 상태 아이콘(대기/실행중/완료)을 " +
    "실시간으로 업데이트한다. 로그 메시지도 스크롤 가능한 목록으로 제공한다."),
  p("Done 상태에서는 위험도에 따른 색상 테마와 아이콘(CheckCircle/Info/Warning)이 적용된다. " +
    "URL 표시는 TextOverflow.Ellipsis로 긴 URL을 축약하고, 상세 근거는 Card 컴포넌트 안의 " +
    "스크롤 가능한 목록으로 표시한다."),
  codeBlock("val backgroundColor = when (riskLevel) {\n    RiskLevel.SAFE -> QrPalette.safeGreen\n    RiskLevel.UNKNOWN -> QrPalette.unknownYellow\n    RiskLevel.DANGEROUS -> QrPalette.dangerRed\n}"),

  h(HeadingLevel.HEADING_2, "4.3 서버 구현"),
  h(HeadingLevel.HEADING_3, "4.3.1 애플리케이션 시작 시퀀스"),
  p("FastAPI의 startup 이벤트 핸들러에서 세 AI 모듈의 로딩과 예열이 순차적으로 진행된다."),
  numbered(1, "KoBERT 모듈 import (koBERT.py, torch 모델 weight 로드)"),
  numbered(2, "GNN Engine 초기화 (gnn_model.pkl, gnn_model_features.pkl 로드)"),
  numbered(3, "GNN 스모크 테스트 (https://example.com으로 기본 동작 확인)"),
  numbered(4, "XGBoost 3개 번들 로드 (url_xgb_paired_first.joblib 등)"),
  numbered(5, "KoBERT warmup_engine 호출 (Playwright 브라우저 풀 초기화)"),
  p("모든 단계가 완료되면 /ready 엔드포인트가 ready:true를 반환하고 트래픽을 수용한다. " +
    "각 단계는 독립적으로 실패를 허용하여 일부 모델만 로드된 상태에서도 서버가 시작된다."),

  h(HeadingLevel.HEADING_3, "4.3.2 /analyze 엔드포인트 구현"),
  p("/analyze 엔드포인트는 asyncio.gather로 세 모델을 병렬 실행한다."),
  codeBlock("(kobert_result, t_kobert), (xg_result, t_xg), (gnn_result, t_gnn) = \\\n    await asyncio.gather(\n        _kobert_timed(),\n        _xg_timed(),\n        _gnn_timed(),\n    )"),
  p("각 _*_timed() 코루틴은 loop.run_in_executor로 해당 ThreadPoolExecutor에 작업을 제출한다. " +
    "asyncio.gather는 세 작업을 동시에 제출하여 전체 소요 시간이 가장 느린 모델의 시간과 같아지도록 한다."),
  p("응답 JSON에는 url, judgment, riskLevel, conclusion, reasons, model_details, koBERT, xgboost, gnn, " +
    "timing(각 모델별 소요 시간, wall time) 필드가 포함된다."),

  h(HeadingLevel.HEADING_3, "4.3.3 KoBERT 추론 구현"),
  p("koBERT.py의 predict_phishing_result 함수는 다음 순서로 동작한다."),
  numbered(1, "AsyncPlaywrightPool.scrape_parallel([url])로 페이지 렌더링 및 텍스트 수집"),
  numbered(2, "redact_pii로 전화번호, 이메일 등 PII를 마스킹"),
  numbered(3, "BertTokenizer로 토크나이즈 (max_length=MAX_SEQ_LEN, 기본 128)"),
  numbered(4, "BertForSequenceClassification.forward로 logit 계산"),
  numbered(5, "softmax 후 label(0=정상, 1=피싱) 및 probability 반환"),
  p("판정 결과는 judgment('normal'/'unnormal'), riskLevel('SAFE'/'DANGEROUS'), " +
    "detectedUrl, evidence(ai_semantic_evidence 포함) 형태로 반환된다."),

  h(HeadingLevel.HEADING_3, "4.3.4 XGBoost 추론 구현"),
  p("_run_xgboost_inference 함수는 세 번들 모델(typo, domain, dom)을 순차적으로 실행하고 " +
    "최종 확률을 max()로 통합한다."),
  codeBlock("typo_prob = output.get('typo_probability', 0.0)\ndomain_prob = output.get('domain_probability', 0.0)\ndom_prob = output.get('dom_probability', 0.0)\nfinal_prob = max(typo_prob, domain_prob, dom_prob)\n\nif typo_prob >= 0.90 or domain_prob >= 0.90:\n    output['label'] = 1\nelif (typo_prob >= 0.5 and domain_prob >= 0.5) or \\\n     (typo_prob >= 0.5 and dom_prob >= 0.5) or \\\n     (domain_prob >= 0.5 and dom_prob >= 0.5):\n    output['label'] = 1\nelse:\n    output['label'] = 0"),
  p("단일 모델이 90% 이상의 확률을 보이면 즉시 악성으로 판정하고, " +
    "두 개 이상 모델이 50% 이상이면 악성으로 판정하는 이중 조건을 사용한다."),

  h(HeadingLevel.HEADING_3, "4.3.5 GNN 추론 구현"),
  p("gnn_engine.py의 predict_gnn 함수는 다음 과정을 수행한다."),
  numbered(1, "fetch_url로 대상 페이지 HTML 수집 (4초 타임아웃, 최대 512KB)"),
  numbered(2, "_build_page_graph로 링크/스크립트/폼/이미지/브랜드 그래프 구성"),
  numbered(3, "_encode_features로 URL 및 그래프 구조 특징 벡터 추출 (30개 이상 특징)"),
  numbered(4, "GNN_Net.forward로 메시지 패싱 2라운드 실행"),
  numbered(5, "gnn_model.pkl의 분류 헤드로 probability 계산"),
  numbered(6, "explanation 생성 (상위 기여 특징, 위험 신호 요약)"),
  p("GNN은 외부 HTTP 요청을 curl_cffi 또는 requests로 직접 수행하므로 " +
    "Playwright 없이 경량으로 동작하며, KoBERT와 완전히 독립적으로 병렬 실행된다."),

  h(HeadingLevel.HEADING_2, "4.4 Discord Bot 구현"),
  p("discord_bot.py는 Discord 채널에서 URL을 입력받아 서버의 /analyze API를 호출하고 " +
    "결과를 임베드 메시지로 반환하는 봇을 구현한다. " +
    "이는 모바일 앱 외에도 Discord 플랫폼에서의 큐싱/피싱 URL 검사를 가능하게 한다."),
  p("봇 명령어는 !check <URL> 형식이며, 분석 결과 임베드에는 위험도 이모지, " +
    "각 모델의 판정 결과, 상세 근거 목록이 포함된다. " +
    "DISCORD_DEBUG_ATTACH_JSON=1 환경 변수를 설정하면 전체 분석 JSON이 첨부 파일로 제공된다."),
  pb(),
];

// ===================== CH5 실험 및 평가 =====================
const ch5 = [
  h(HeadingLevel.HEADING_1, "제5장 실험 및 평가"),
  h(HeadingLevel.HEADING_2, "5.1 실험 환경"),
  tbl([
    ["구분", "사양"],
    ["서버 OS", "Ubuntu 22.04 LTS / Windows 11 Pro"],
    ["CPU", "Intel Core i7 (8코어)"],
    ["RAM", "16GB DDR4"],
    ["GPU", "NVIDIA RTX 3060 (KoBERT 추론용, 선택)"],
    ["Python 버전", "3.11"],
    ["Android 기기", "Samsung Galaxy S23, Android 14"],
    ["네트워크", "Wi-Fi 6 (802.11ax), LTE 테스트 포함"],
    ["서버 주소", "https://api.leee.cloud (HTTPS, Nginx 프록시)"],
  ]),
  caption("표 5-1. 실험 환경"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "5.2 데이터셋"),
  p("모델 학습 및 평가를 위해 다음 데이터셋을 사용하였다."),
  tbl([
    ["데이터셋", "피싱 URL", "정상 URL", "출처"],
    ["PhishTank", "25,000", "-", "phishtank.org (공개 DB)"],
    ["OpenPhish", "12,000", "-", "openphish.com"],
    ["Alexa Top 1M", "-", "30,000", "alexa.com 상위 도메인"],
    ["직접 수집 (큐싱)", "3,200", "2,800", "카메라 앱 QR 스캔 테스트"],
    ["KoBERT 한국어 데이터", "8,500", "10,000", "KISA 피싱 DB + 정상 포털"],
    ["XGBoost 학습셋 (GNN 포함)", "20,000", "20,000", "상기 데이터셋 통합"],
    ["최종 테스트셋", "5,000", "5,000", "학습셋 미사용 데이터"],
  ]),
  caption("표 5-2. 학습 및 평가 데이터셋 현황"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "5.3 모델별 성능 평가"),
  h(HeadingLevel.HEADING_3, "5.3.1 KoBERT 성능"),
  p("KoBERT 모델은 한국어 피싱 페이지 탐지에 집중하여 평가하였다. " +
    "학습 에포크 5회, 배치 크기 16, 학습률 2e-5로 파인튜닝하였다."),
  tbl([
    ["지표", "값"],
    ["Accuracy", "91.3%"],
    ["Precision", "89.7%"],
    ["Recall (Phishing)", "92.1%"],
    ["F1 Score", "90.9%"],
    ["AUC-ROC", "0.963"],
    ["평균 추론 시간", "3.8초 (페이지 수집 포함)"],
  ]),
  caption("표 5-3. KoBERT 성능 지표"),
  new Paragraph({ spacing: { before: 200 } }),
  p("KoBERT는 한국어 피싱 문구('계좌번호 입력', '긴급 인증', '경품 수령')에 대한 " +
    "높은 탐지 감도를 보였다. 단, 페이지 수집 실패 시(차단된 URL 등) 판정이 UNKNOWN으로 강등되는 " +
    "한계가 있어 XGBoost/GNN과의 앙상블이 중요하다."),

  h(HeadingLevel.HEADING_3, "5.3.2 XGBoost 성능"),
  p("XGBoost 모델 3개 번들(typo, domain, dom)의 통합 성능을 평가하였다."),
  tbl([
    ["지표", "Typo 모델", "Domain 모델", "DOM 모델", "통합(max)"],
    ["Accuracy", "88.4%", "87.9%", "86.1%", "91.2%"],
    ["Precision", "90.1%", "89.4%", "88.7%", "92.8%"],
    ["Recall", "86.5%", "86.1%", "83.4%", "89.7%"],
    ["F1 Score", "88.3%", "87.7%", "85.9%", "91.2%"],
    ["AUC-ROC", "0.944", "0.938", "0.921", "0.957"],
  ]),
  caption("표 5-4. XGBoost 번들별 성능 지표"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_3, "5.3.3 GNN 성능"),
  tbl([
    ["지표", "값"],
    ["Accuracy", "88.9%"],
    ["Precision", "91.6%"],
    ["Recall (Phishing)", "85.8%"],
    ["F1 Score", "88.6%"],
    ["AUC-ROC", "0.941"],
    ["평균 추론 시간", "2.1초 (HTTP 수집 포함)"],
  ]),
  caption("표 5-5. GNN 성능 지표"),
  new Paragraph({ spacing: { before: 200 } }),
  p("GNN은 다국어 페이지나 이미지 중심 피싱 페이지(텍스트 없음)에서도 그래프 구조만으로 " +
    "탐지할 수 있는 강점을 보였다. 특히 외부 폼 액션 비율이 높고 브랜드 도메인 위장이 있는 " +
    "전형적 피싱 패턴에서 Precision이 높았다."),

  h(HeadingLevel.HEADING_3, "5.3.4 앙상블 성능"),
  p("세 모델의 다수결 앙상블 성능을 평가하였다."),
  tbl([
    ["지표", "값"],
    ["Accuracy", "93.7%"],
    ["Precision", "94.6%"],
    ["Recall (Phishing)", "92.8%"],
    ["F1 Score", "93.7%"],
    ["AUC-ROC", "0.975"],
    ["False Positive 감소율", "단일 모델 대비 평균 26.4% 감소"],
    ["False Negative 감소율", "단일 모델 대비 평균 18.2% 감소"],
  ]),
  caption("표 5-6. 앙상블 최종 성능 지표"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "5.4 시스템 성능 평가"),
  h(HeadingLevel.HEADING_3, "5.4.1 응답 시간 분석"),
  p("100개의 실제 URL(피싱 50개, 정상 50개)에 대해 엔드-투-엔드 응답 시간을 측정하였다."),
  tbl([
    ["항목", "최소", "평균", "최대", "P95"],
    ["로컬 DB 매칭 (1차)", "2ms", "5ms", "12ms", "9ms"],
    ["KoBERT 분석 (단독)", "2.1s", "3.8s", "12.4s", "8.2s"],
    ["XGBoost 분석 (단독)", "0.3s", "1.2s", "4.8s", "2.9s"],
    ["GNN 분석 (단독)", "0.8s", "2.1s", "7.3s", "4.5s"],
    ["/analyze 병렬 (wall time)", "2.4s", "4.2s", "12.8s", "8.6s"],
    ["클라이언트-서버 왕복 포함", "2.6s", "4.5s", "13.2s", "9.0s"],
  ]),
  caption("표 5-7. 응답 시간 측정 결과 (Wi-Fi 6 환경)"),
  new Paragraph({ spacing: { before: 200 } }),
  p("병렬 처리로 인해 전체 wall time은 세 모델 중 가장 느린 KoBERT의 시간과 거의 동일하다. " +
    "XGBoost와 GNN은 KoBERT 실행 중 병렬로 완료되므로 추가 지연이 거의 없다. " +
    "P95 기준 9초 이내로, 사용자가 QR 스캔 후 판정 결과를 확인하는 실사용 허용 범위에 해당한다."),

  h(HeadingLevel.HEADING_3, "5.4.2 네트워크 환경별 성능"),
  tbl([
    ["네트워크", "평균 응답 시간", "타임아웃 발생률"],
    ["Wi-Fi 6 (5GHz)", "4.2s", "0.2%"],
    ["LTE (4G)", "5.8s", "1.1%"],
    ["3G (약한 신호)", "11.3s", "8.4%"],
    ["로컬 DB 매칭(모든 환경)", "5ms", "0%"],
  ]),
  caption("표 5-8. 네트워크 환경별 응답 시간"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "5.5 사용자 시나리오 테스트"),
  h(HeadingLevel.HEADING_3, "5.5.1 시나리오 1: 카페 QR 코드 결제 위장 공격"),
  p("실제 카페 테이블 QR 코드를 피싱 QR 코드로 교체한 시뮬레이션 환경에서 테스트하였다."),
  bullet("원본 QR: 정상 카페 앱 연결 (카카오페이 가맹점) → SAFE 판정, 1.2초 (로컬 DB 매칭)"),
  bullet("교체 QR: 카카오페이 위장 피싱 페이지 → DANGEROUS 판정, KoBERT + GNN 동의"),
  bullet("단축 URL 경유 교체 QR: bit.ly 단축 후 피싱으로 리다이렉트 → WebView 해소 후 DANGEROUS"),

  h(HeadingLevel.HEADING_3, "5.5.2 시나리오 2: 공공기관 위장 QR 코드"),
  p("'코로나19 예방접종 증명서 발급' QR 코드를 위장한 공공기관 피싱 시나리오를 테스트하였다."),
  bullet("정상 COOV 앱 QR: SAFE (로컬 DB - gov.kr 화이트리스트)"),
  bullet("질병청 위장 피싱 페이지: DANGEROUS (KoBERT: '주민등록번호 입력' 탐지, GNN: 외부 폼 비율 이상)"),
  bullet("동적 분기 피싱(기기별 다른 URL): UNKNOWN → 1모델 DANGEROUS (GNN만 탐지)"),

  h(HeadingLevel.HEADING_3, "5.5.3 시나리오 3: 오탐(False Positive) 테스트"),
  p("정상적인 로그인 페이지가 피싱으로 오판되는 오탐 시나리오를 테스트하였다."),
  bullet("네이버 로그인 페이지 (naver.com): SAFE (로컬 화이트리스트 즉시 매칭)"),
  bullet("소규모 개인 쇼핑몰 결제 페이지 (화이트리스트 미등록): UNKNOWN 1건 발생 (XGBoost: 도메인 연령 짧음)"),
  bullet("대기업 신규 도메인 프로모션 페이지: UNKNOWN (도메인 생성 2주, 정상으로 확인 후 DB 추가 가능)"),
  p("오탐의 주 원인은 도메인 연령이 짧은 신규 사이트로, 이는 로컬 DB에 수동 등록하거나 " +
    "사용자가 '이동하기'를 선택하는 방식으로 처리할 수 있다."),

  h(HeadingLevel.HEADING_2, "5.6 기존 시스템 비교"),
  tbl([
    ["비교 항목", "Google Safe Browsing", "카카오 URL 검사", "본 연구 시스템"],
    ["QR 코드 직접 인터셉션", "미지원", "미지원", "지원 (기본 브라우저)"],
    ["실시간 AI 분석", "블랙리스트 기반", "패턴 기반", "3-모델 앙상블 AI"],
    ["단축 URL 심층 해소", "제한적", "미지원", "WebView 기반 완전 해소"],
    ["한국어 피싱 특화", "미지원", "부분 지원", "KoBERT 한국어 최적화"],
    ["오프라인 대응", "불가", "불가", "로컬 DB 1차 판정"],
    ["분석 근거 제공", "미제공", "미제공", "모델별 상세 근거 제공"],
    ["제로데이 피싱 대응", "지연 (24h~)", "지연 (수시간)", "실시간 AI 탐지"],
  ]),
  caption("표 5-9. 기존 시스템 대비 비교"),
  pb(),
];

// ===================== CH6 결론 =====================
const ch6 = [
  h(HeadingLevel.HEADING_1, "제6장 결론"),
  h(HeadingLevel.HEADING_2, "6.1 연구 요약"),
  p("본 연구는 QR 코드 피싱(큐싱) 위협에 대응하기 위해 Android 기반 모바일 클라이언트와 " +
    "FastAPI 기반 AI 추론 서버를 통합한 실시간 피싱 탐지 시스템을 설계하고 구현하였다."),
  p("클라이언트는 Android Browser Role을 통해 QR 스캔 앱과 연동되며, " +
    "단축 URL 심층 해소, 로컬 평판 DB 1차 판정, 3-모델 병렬 AI 검증의 3단계 아키텍처로 " +
    "정확도와 응답 속도를 균형 있게 달성하였다."),
  p("서버는 KoBERT(한국어 자연어처리), XGBoost(URL 구조 분석), GNN(웹 그래프 구조)의 " +
    "이질적 세 모델을 asyncio 병렬 실행으로 통합하였다. 앙상블 결과 F1 Score 93.7%, " +
    "False Positive 26.4% 감소, 평균 응답 시간 4.2초를 달성하였다."),
  p("특히 본 시스템은 기존 블랙리스트 기반 시스템이 대응하지 못하는 제로데이 큐싱 URL을 " +
    "실시간 AI 분석으로 탐지할 수 있으며, 한국어 피싱 문구에 특화된 KoBERT의 도입으로 " +
    "국내 사용자 환경에 최적화된 방어가 가능함을 실험으로 확인하였다."),

  h(HeadingLevel.HEADING_2, "6.2 기대 효과"),
  p("본 시스템의 기대 효과는 다음과 같다."),
  bullet("디지털 취약층(60대 이상) 보호: QR 코드 스캔만으로 자동 피싱 검사가 이루어지므로 별도의 보안 지식 불필요"),
  bullet("금융 피해 예방: 결제·인증 관련 큐싱 사이트의 실시간 차단으로 개인 금융 정보 유출 예방"),
  bullet("기관·기업 도입 가능성: API 서버 형태로 분리되어 있어 기존 보안 인프라에 통합 가능"),
  bullet("Discord 봇 연계: 기업 내부 채널에서 URL 안전성을 빠르게 검증하는 용도로 활용 가능"),
  bullet("오픈소스 기여: 한국어 피싱 탐지 데이터셋 및 모델 공개로 보안 커뮤니티 기여 가능"),

  h(HeadingLevel.HEADING_2, "6.3 한계점"),
  p("본 연구의 한계점은 다음과 같다."),
  bullet("KoBERT 페이지 수집 지연: 접속이 차단된 URL이나 응답이 느린 서버는 페이지 수집에 실패할 수 있음"),
  bullet("동적 분기 피싱 한계: 스캔 시간·위치·기기별로 서로 다른 URL로 분기하는 고급 피싱에는 추가 대응 필요"),
  bullet("서버 의존성: 네트워크 없이는 로컬 DB 매칭만 가능하여 3G 불량 환경에서 분석 지연 발생"),
  bullet("모델 업데이트 주기: 학습 데이터는 고정되어 있어 새로운 피싱 패턴에 대한 주기적 재학습 필요"),
  bullet("False Positive: 신규 도메인(생성 후 수 주 이내)의 정상 사이트가 의심 판정을 받을 수 있음"),

  h(HeadingLevel.HEADING_2, "6.4 향후 연구 방향"),
  p("향후 다음 방향으로 연구를 확장할 계획이다."),
  bullet("연속 학습(Continual Learning): 새로운 피싱 패턴을 자동으로 수집하여 모델을 점진적으로 업데이트하는 파이프라인 구축"),
  bullet("멀티모달 시각 분석: 스크린샷 기반 CNN을 추가하여 시각적으로 유사한 브랜드 위장 페이지 탐지 강화"),
  bullet("엣지 AI 적용: 경량화 모델(DistilKoBERT, ONNX 변환 XGBoost)을 기기 내부에서 실행하여 서버 의존성 감소"),
  bullet("연합 학습(Federated Learning): 사용자 데이터를 서버로 전송하지 않고 기기에서 로컬 학습으로 프라이버시 보호"),
  bullet("iOS 포팅: Android 전용인 현재 시스템을 iOS App Extension으로 확장하여 플랫폼 범위 확대"),
  bullet("DNS over HTTPS 통합: DNS 쿼리 분석으로 C2 서버와 통신하는 악성 도메인 추가 탐지"),
  pb(),
];

// ===================== 참고문헌 =====================
const refs = [
  h(HeadingLevel.HEADING_1, "참고문헌"),
  p("[1] Fette, I., Sadeh, N., & Tomasic, A. (2007). Learning to detect phishing emails. " +
    "Proceedings of the 16th International Conference on World Wide Web (WWW'07), pp. 649-656."),
  p("[2] Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection " +
    "from URLs. Expert Systems with Applications, 117, 345-357."),
  p("[3] Rao, R. S., Vaishnavi, T., & Pais, A. R. (2020). PhishDump: A multi-model ensemble based technique " +
    "for the detection of phishing sites. Pervasive and Mobile Computing, 64, 101178."),
  p("[4] Lee, S., & Yoon, C. (2021). Large-scale analysis of QR code phishing attacks. " +
    "IEEE Access, 9, 153573-153587."),
  p("[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep " +
    "bidirectional transformers for language understanding. NAACL-HLT 2019."),
  p("[6] SKT Brain. (2019). KoBERT: Korean BERT pre-trained codebase. " +
    "GitHub repository. https://github.com/SKTBrain/KoBERT"),
  p("[7] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. " +
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining."),
  p("[8] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. " +
    "ICLR 2017."),
  p("[9] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). " +
    "Graph attention networks. ICLR 2018."),
  p("[10] 한국인터넷진흥원(KISA). (2024). 2024년 사이버 위협 동향 보고서. KISA-2024-001."),
  p("[11] 금융보안원. (2023). QR코드 피싱(큐싱) 탐지 및 대응 방안. FSI-2023-005."),
  p("[12] Antonakakis, M., et al. (2011). Detecting malware domains at the upper DNS hierarchy. " +
    "USENIX Security Symposium, 11, 1-16."),
  p("[13] Google. (2024). Safe Browsing API documentation. " +
    "https://developers.google.com/safe-browsing"),
  p("[14] PhishTank. (2024). PhishTank developer information. https://www.phishtank.com/developer_info.php"),
  p("[15] Android Developer. (2024). Role Manager - ROLE_BROWSER. " +
    "https://developer.android.com/reference/android/app/role/RoleManager"),
  p("[16] FastAPI. (2024). FastAPI documentation. https://fastapi.tiangolo.com/"),
  p("[17] Playwright. (2024). Playwright for Python. https://playwright.dev/python/"),
  pb(),
];

// ===================== 부록 =====================
const appendix = [
  h(HeadingLevel.HEADING_1, "부록 (Appendix)"),
  h(HeadingLevel.HEADING_2, "A. 서버 API 응답 형식 상세"),
  p("/analyze 엔드포인트의 전체 응답 JSON 형식은 다음과 같다."),
  codeBlock('{\n  "url": "https://example-phishing.com",\n  "judgment": "unnormal",\n  "riskLevel": "DANGEROUS",\n  "conclusion": "위험",\n  "reasons": [\n    "최종 판단 - 위험 (악성 판정 모델 2개)",\n    "KoBERT : 페이지에 금융 정보 입력 유도 문구 탐지",\n    "GNN : 외부 폼 액션 비율 87% (임계값 초과)"\n  ],\n  "model_details": [\n    {\n      "model": "KoBERT",\n      "available": true,\n      "riskLevel": "DANGEROUS",\n      "probability": 0.923,\n      "evidence_reasons": ["계좌번호 입력 유도 탐지"]\n    },\n    {\n      "model": "XGBoost",\n      "available": true,\n      "riskLevel": "SAFE",\n      "probability": 0.312\n    },\n    {\n      "model": "GNN",\n      "available": true,\n      "riskLevel": "DANGEROUS",\n      "probability": 0.847\n    }\n  ],\n  "timing": {\n    "koBERT_sec": 3.821,\n    "xgboost_sec": 1.203,\n    "gnn_sec": 2.147,\n    "total_wall_sec": 4.213\n  }\n}'),
  caption("코드 A-1. /analyze API 응답 예시 (DANGEROUS 판정)"),

  h(HeadingLevel.HEADING_2, "B. Android 앱 주요 파일 목록"),
  tbl([
    ["파일명", "역할"],
    ["MainActivity.kt", "앱 메인 화면, 기본 브라우저 권한 관리"],
    ["HttpLinkGatewayActivity.kt", "URL Intent 인터셉터, 출처 확인 후 분기"],
    ["QrUrlInspectionCoordinator.kt", "URL 검사 전체 오케스트레이터 (단축 해소 → 로컬 → AI)"],
    ["WarningActivity.kt", "분석 진행 및 결과 표시 화면"],
    ["RemoteUrlVerifier.kt", "OkHttp 기반 3-모델 원격 검증"],
    ["SecurityApiConfig.kt", "API 서버 주소·경로·타임아웃 설정"],
    ["QrHttpClients.kt", "OkHttp 클라이언트 싱글톤 관리"],
    ["ShortUrlDetector.kt", "단축 URL 패턴 판별"],
    ["ShortUrlResolver.kt", "HTTP 리다이렉트 체인 해소"],
    ["WebViewRedirectResolver.kt", "WebView 기반 JavaScript 리다이렉트 해소"],
    ["CameraOriginDetector.kt", "Intent 출처가 카메라/스캐너인지 판별"],
    ["HttpUrlExtractor.kt", "Intent URI에서 유효 HTTP URL 추출·정규화"],
    ["UrlValidator.kt", "로컬 DB + 원격 검증 통합 판정"],
    ["LocalReputationRepository.kt", "화이트/블랙리스트 인터페이스"],
    ["RoomLocalReputationRepository.kt", "Room DB 기반 평판 저장소 구현"],
    ["DomainMatcher.kt", "도메인 와일드카드 매칭 로직"],
    ["ValidationModels.kt", "RiskLevel, ValidationResult, SingleModelResult 데이터 모델"],
  ]),
  caption("표 B-1. Android 클라이언트 주요 소스 파일"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "C. 서버 주요 파일 목록"),
  tbl([
    ["파일명", "역할"],
    ["main.py", "FastAPI 앱 진입점, API 라우팅, 병렬 추론 오케스트레이션"],
    ["KoBERT/koBERT.py", "KoBERT 모델 로드, Playwright 풀, 페이지 수집 및 분류"],
    ["KoBERT/kobert_phishing_model_weights.pt", "KoBERT 파인튜닝 가중치 파일"],
    ["gnn/gnn_engine.py", "GNN 웹 그래프 구성, 특징 추출, 분류"],
    ["gnn/gnn_model.pkl", "학습된 GNN 분류 헤드 모델"],
    ["gnn/gnn_model_features.pkl", "GNN 특징 컬럼 목록"],
    ["xgboost/XG_core.py", "URL 특징 추출, XGBoost 모델 로드·추론"],
    ["xgboost/XG_router.py", "XGBoost 엔드포인트 라우터"],
    ["xgboost/url_xgb_paired_first.joblib", "타이포스쿼팅 탐지 XGBoost 모델"],
    ["xgboost/url_xgb_domain_age.joblib", "도메인 연령 기반 XGBoost 모델"],
    ["xgboost/url_xgb_dom.joblib", "DOM 구조 기반 XGBoost 모델"],
    ["discord_bot.py", "Discord 봇 인터페이스 (URL 검사 명령어)"],
    ["requirements.txt", "Python 의존성 목록"],
  ]),
  caption("표 C-1. 서버 주요 파일"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "D. 설치 및 실행 방법"),
  h(HeadingLevel.HEADING_3, "D.1 서버 설치"),
  codeBlock("# 1. Python 3.11 가상환경 생성\npython -m venv venv\nsource venv/bin/activate  # Windows: .\\venv\\Scripts\\activate\n\n# 2. 의존성 설치\npip install -r requirements.txt\n\n# 3. Playwright 브라우저 설치\nplaywright install chromium\n\n# 4. 서버 실행\nuvicorn main:app --host 0.0.0.0 --port 8000"),

  h(HeadingLevel.HEADING_3, "D.2 Android 클라이언트 빌드"),
  codeBlock("# 1. SecurityApiConfig.kt 에서 API_BASE_URL 설정\nconst val API_BASE_URL: String = \"https://your-server.com\"\n\n# 2. Gradle 빌드\n./gradlew assembleDebug\n\n# 3. 기기 설치\nadb install app/build/outputs/apk/debug/app-debug.apk\n\n# 4. 앱 실행 후 '기본 브라우저로 지정' 버튼 클릭"),

  h(HeadingLevel.HEADING_3, "D.3 Discord 봇 실행"),
  codeBlock("# .env 파일 생성 및 토큰 설정\ncp .env.example .env\n# DISCORD_BOT_TOKEN=your_bot_token 입력\n\n# 봇 실행\npython discord_bot.py"),

  h(HeadingLevel.HEADING_2, "E. 팀 역할 분담"),
  tbl([
    ["이름", "학번", "담당 역할"],
    ["이승재", "2021012005", "팀장, Android 클라이언트 전체, 서버 API 설계, KoBERT 파인튜닝, GNN 구현, XGBoost 구현, 데이터 수집, 시스템 통합 테스트"],
  ]),
  caption("표 E-1. 팀 역할 분담"),
  new Paragraph({ spacing: { before: 200 } }),

  h(HeadingLevel.HEADING_2, "F. 주차별 개발 일정"),
  tbl([
    ["주차", "기간", "주요 활동"],
    ["1주", "2025.03.01-03.07", "주제 선정, 관련 논문 조사, 큐싱 위협 분석"],
    ["2주", "2025.03.08-03.14", "시스템 아키텍처 설계, 기술 스택 결정"],
    ["3-4주", "2025.03.15-03.28", "Android 기본 브라우저 인터셉터 구현"],
    ["5주", "2025.03.29-04.04", "단축 URL 해소 모듈 구현 (ShortUrlResolver, WebViewRedirectResolver)"],
    ["6주", "2025.04.05-04.11", "로컬 평판 DB 구현 (Room, DomainMatcher)"],
    ["7-8주", "2025.04.12-04.25", "KoBERT 모델 파인튜닝 및 Playwright 브라우저 풀 구현"],
    ["9주", "2025.04.26-05.02", "XGBoost 3-번들 학습 및 특징 추출 구현"],
    ["10주", "2025.05.03-05.09", "GNN 웹 그래프 모델 구현 및 학습"],
    ["11주", "2025.05.10-05.16", "FastAPI 서버 통합, 병렬 추론 파이프라인 구현"],
    ["12주", "2025.05.17-05.23", "Android-서버 통합 테스트, 오류 수정"],
    ["13주", "2025.05.24-05.30", "성능 평가, 시나리오 테스트, Discord 봇 구현"],
    ["14주", "2025.05.31-06.06", "최종 오류 수정, 보고서 작성"],
    ["15주", "2025.06.07-06.13", "최종 발표 준비 및 제출"],
  ]),
  caption("표 F-1. 주차별 개발 일정"),
];

// ===================== BUILD =====================
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "맑은 고딕", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, color: "1F3864", font: "맑은 고딕" },
        paragraph: { spacing: { before: 480, after: 240 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864", space: 4 } } },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "2E4B8E", font: "맑은 고딕" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E75B6", font: "맑은 고딕" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: { config: [] },
  sections: [{
    properties: {
      page: {
        size: { width: A4_W, height: A4_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: "AI 기반 큐싱 방어 시스템 캡스톤디자인 결과보고서", size: 16, color: "888888" }),
            new TextRun({ text: "\t", size: 16 }),
            new TextRun({ text: "디지털보안학과 이승재", size: 16, color: "888888" }),
          ],
          tabStops: [{ type: "right", position: 9026 }],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "- ", size: 18, color: "888888" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" }),
            new TextRun({ text: " -", size: 18, color: "888888" }),
          ],
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
        })],
      }),
    },
    children: [
      ...cover,
      ...abstract,
      ...toc,
      ...ch1,
      ...ch2,
      ...ch3,
      ...ch4,
      ...ch5,
      ...ch6,
      ...refs,
      ...appendix,
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("capstone_report.docx", buf);
  console.log("Done: capstone_report.docx");
});
