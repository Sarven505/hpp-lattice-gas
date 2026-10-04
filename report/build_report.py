"""Build the Russian laboratory PDF and its Markdown source from measured results."""

import argparse
import io
import json
import keyword
import re
import tokenize
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Sarven505/hpp-lattice-gas"
TITLE = "Исследование модели HPP: перераспределение плотности газа и влияние граничных условий"


def build(output: Path):
    summary = json.loads((ROOT / "experiments/summary.json").read_text())
    b = summary["boundaries"]
    config = summary["config"]
    if config != dict(width=128, height=96, steps=800, seeds=12, block=8, p_left=0.45,
                      p_right=0.05, seed_start=2026, threshold=0.1, window=25):
        raise ValueError("Этот текст отчёта относится только к сохранённому эксперименту 128 × 96, 800 шагов, 12 seed.")
    fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [("Body", "DejaVuSerif.ttf"), ("BodyBold", "DejaVuSerif-Bold.ttf"),
                       ("Sans", "DejaVuSans.ttf"), ("SansBold", "DejaVuSans-Bold.ttf"),
                       ("Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fontdir / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    styles = {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=15,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "head": ParagraphStyle("head", fontName="SansBold", fontSize=14, leading=19,
                               spaceAfter=12, textColor=colors.HexColor("#244b64")),
        "sub": ParagraphStyle("sub", fontName="SansBold", fontSize=11, leading=15,
                              spaceBefore=7, spaceAfter=7),
        "center": ParagraphStyle("center", fontName="Body", fontSize=11, leading=16,
                                 alignment=TA_CENTER, spaceAfter=5),
        "title": ParagraphStyle("title", fontName="SansBold", fontSize=16, leading=23,
                                alignment=TA_CENTER, spaceAfter=12),
        "small": ParagraphStyle("small", fontName="Sans", fontSize=8, leading=11,
                                spaceAfter=6),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=9, leading=13,
                                  alignment=TA_LEFT, spaceAfter=12),
        "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.5, leading=12),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=7.8, leading=11),
        "right": ParagraphStyle("right", fontName="Body", fontSize=10, alignment=TA_RIGHT),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    story, md = [], [f"# {TITLE}\n", "Растегаев Алексей Эдуардович, СМ6-91.\n",
                     "МГТУ им. Н. Э. Баумана; кафедра СМ6; Компьютерное моделирование.\n",
                     "Преподаватель: Федулов В. А. Москва, 2026.\n", f"Исходный код: {REPO}\n"]
    temporary = TemporaryDirectory()
    tmp = Path(temporary.name)
    eqcounter = 0

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))
        md.append(text.replace("<b>", "**").replace("</b>", "**") + "\n")

    def heading(text, newpage=True):
        if newpage: story.append(PageBreak())
        p(text, "head")
        md[-1] = "## " + md[-1]

    def sub(text):
        p(text, "sub"); md[-1] = "### " + md[-1]

    def eq(text, height=12):
        nonlocal eqcounter
        eqcounter += 1
        fig = plt.figure(figsize=(8, height / 25.4), dpi=220)
        fig.text(0.5, 0.5, "$" + text + "$", ha="center", va="center", fontsize=15)
        path = tmp / f"eq{eqcounter}.png"
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        im = Image(str(path))
        scale = min(145*mm / im.imageWidth, height*mm / im.imageHeight)
        im.drawWidth, im.drawHeight = im.imageWidth*scale, im.imageHeight*scale
        table = Table([[im, Paragraph(f"({eqcounter})", styles["right"])]], colWidths=[154*mm, 12*mm])
        table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(table)
        md.append(f"$$ {text} \\quad ({eqcounter}) $$\n")

    def table(caption, data, widths):
        p(caption, "caption")
        cells = [[Paragraph(escape(str(x)), styles["cell"]) for x in row] for row in data]
        t = Table(cells, colWidths=[x*mm for x in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf2f6")),
                               ("LINEBELOW",(0,0),(-1,0),0.7,colors.HexColor("#6b8798")),
                               ("LINEBELOW",(0,1),(-1,-1),0.3,colors.HexColor("#d7dfe4")),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("TOPPADDING",(0,0),(-1,-1),7),
                               ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        story.append(t); story.append(Spacer(1,8))
        md.extend(["| " + " | ".join(str(x).replace("|", r"\|") for x in row) + " |\n" for row in data[:1]])
        md.append("|" + "---|"*len(data[0]) + "\n")
        md.extend(["| " + " | ".join(str(x).replace("|", r"\|") for x in row) + " |\n" for row in data[1:]])

    def figure(name, caption, width=166):
        path = ROOT / "experiments/figures" / name
        im = Image(str(path))
        im.drawHeight = im.imageHeight / im.imageWidth * width*mm
        im.drawWidth = width*mm
        story.append(KeepTogether([im, Spacer(1,5), Paragraph(caption, styles["caption"])]))
        md.extend([f"![{caption}](../experiments/figures/{name})\n"])

    def code(text):
        lines = text.strip().splitlines()
        markup = [escape(line).replace(" ", "&#160;") for line in lines]
        # Colour token spans without changing the code's text or indentation.
        tokens_by_line = {}
        for token in tokenize.generate_tokens(io.StringIO(text.strip()).readline):
            color = None
            if token.type == tokenize.NAME and keyword.iskeyword(token.string): color = "#6f42a1"
            elif token.type == tokenize.STRING: color = "#23734c"
            elif token.type == tokenize.COMMENT: color = "#64748b"
            if color and token.start[0] == token.end[0]:
                tokens_by_line.setdefault(token.start[0]-1, []).append((token.start[1],token.end[1],color))
        formatted = []
        for i, line in enumerate(lines):
            pos, spans = 0, []
            for start,end,color in tokens_by_line.get(i,[]):
                spans += [escape(line[pos:start]).replace(" ", "&#160;"),
                          f'<font color="{color}">{escape(line[start:end]).replace(" ", "&#160;")}</font>']
                pos = end
            spans.append(escape(line[pos:]).replace(" ", "&#160;"))
            formatted.append(Paragraph("".join(spans), styles["code"]))
        t = Table([[line] for line in formatted], colWidths=[166*mm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f5f7fa")),
                              ("TOPPADDING",(0,0),(-1,-1),1), ("BOTTOMPADDING",(0,0),(-1,-1),1)]))
        story.append(KeepTogether([t,Spacer(1,8)]))
        md.append("```python\n" + text.strip() + "\n```\n")

    def fmt(v): return f"{v:.6f}"

    p("Министерство науки и высшего образования Российской Федерации", "center")
    p("Федеральное государственное автономное образовательное учреждение высшего образования", "center")
    p("«Московский государственный технический университет имени Н. Э. Баумана<br/>(национальный исследовательский университет)»", "center")
    story.append(Spacer(1,12*mm))
    p("ФАКУЛЬТЕТ «СПЕЦИАЛЬНОЕ МАШИНОСТРОЕНИЕ»", "center")
    p("КАФЕДРА «РАКЕТНЫЕ И ИМПУЛЬСНЫЕ СИСТЕМЫ» (СМ6)", "center")
    story.append(Spacer(1,20*mm))
    p("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ", "title")
    p("По дисциплине «Компьютерное моделирование»", "center")
    story.append(Spacer(1,8*mm))
    p(TITLE, "title")
    story.append(Spacer(1,22*mm))
    p("Выполнил: студент группы СМ6-91<br/><b>Растегаев Алексей Эдуардович</b>", "body")
    p("Проверил: <b>Федулов В. А.</b>", "body")
    story.append(Spacer(1,15*mm))
    p("Москва, 2026 г.", "center")

    md = [f"# {TITLE}\n", "**Растегаев Алексей Эдуардович, СМ6-91.**\n",
          "МГТУ им. Н. Э. Баумана, кафедра СМ6. Компьютерное моделирование.\n",
          "Преподаватель: Федулов В. А. Москва, 2026.\n"]

    heading("1. Цель работы и модель HPP")
    p("<b>Цель:</b> изучить перенос газа из более плотной половины области в менее плотную и сравнить периодические и отражающие границы. Для этого реализована модель HPP, выполнены повторные расчёты и проверены законы сохранения.")
    p("Поле состоит из H × W узлов. Состояние - булев массив n(y, x, i) размера (H, W, 4). Четыре канала i=0,1,2,3 соответствуют востоку, северу, западу и югу со скоростями (1,0), (0,1), (-1,0), (0,-1). В каждом канале допускается одна частица. Используется окрестность фон Неймана.")
    sub("Столкновение и перенос")
    p("Один шаг состоит из двух действий. Сначала ровно две встречные частицы меняют направления на перпендикулярные. Затем все частицы одновременно перемещаются на один узел. При наличии третьей или четвёртой частицы столкновение не меняет состояние.")
    eq(r"n_i^*=n_i-n_i n_{i+2}(1-n_{i+1})(1-n_{i+3})+(1-n_i)(1-n_{i+2})n_{i+1}n_{i+3}", 14)
    p("Индексы в формуле (1) берутся по модулю 4; звёздочка означает состояние после столкновения.")
    table("Таблица 1. Правила для всех 16 состояний узла", [
        ["До столкновения", "После столкновения", "Вариантов"],
        ["Восток и запад, остальные каналы пусты", "Север и юг", "1"],
        ["Север и юг, остальные каналы пусты", "Восток и запад", "1"],
        ["Все остальные состояния", "Без изменений", "14"]], [78,64,24])
    eq(r"n_i(\mathbf{r}+\mathbf{c}_i,t+1)=n_i^*(\mathbf{r},t)", 10)
    p("При периодических границах координаты берутся по модулю W и H: частица появляется с противоположной стороны. При отражении она остаётся в крайнем узле и меняет направление на обратное. Все изменения вычисляются из старого массива, поэтому обновление синхронно.")
    p("Шаг решётки и один такт приняты за единицы. Внешних сил и добавленного затухания нет. Случайно задаётся только начальное состояние; дальнейшее движение детерминировано.")

    heading("2. Измерения и условия эксперимента")
    p("Плотность ρ - число частиц в узле, j - его импульс. Полное число частиц N и суммарный импульс P получаются суммированием по полю:")
    eq(r"\rho=\sum_i n_i,\quad \mathbf{j}=\sum_i n_i\mathbf{c}_i,\quad N=\sum_{\mathbf{r}}\rho,\quad \mathbf{P}=\sum_{\mathbf{r}}\mathbf{j}", 11)
    eq(r"N(t)=N(0),\qquad \mathbf{P}(t)-\mathbf{P}(0)=\sum_{\tau=0}^{t-1}\mathbf{I}_{\mathrm{wall}}(\tau)", 14)
    p("Столкновения и перенос сохраняют число частиц. При периодических границах импульс постоянен. При отражении его изменение равно импульсу от стенок: одно отражение даёт -2cᵢ. Балансы проверяются точно, целыми числами, на каждом шаге.")
    p("Карты усредняются по блокам B размером 8 × 8. Скорость блока - его суммарный импульс, делённый на число частиц; для пустого блока она равна нулю:")
    eq(r"\rho_B=\frac{\sum_{\mathbf{r}\in B}\rho}{|B|},\qquad \mathbf{u}_B=\frac{\sum_{\mathbf{r}\in B}\mathbf{j}}{\sum_{\mathbf{r}\in B}\rho}", 18)
    eq(r"\Delta\rho=\overline{\rho}_L-\overline{\rho}_R,\qquad d(t)=\frac{\Delta\rho(t)}{\Delta\rho(0)}", 10)
    eq(r"CV_\rho=\frac{\mathrm{std}_B(\rho_B)}{\mathrm{mean}_B(\rho_B)},\quad u_{\mathrm{RMS}}=\sqrt{\frac{1}{K}\sum_B|\mathbf{u}_B|^2}", 15)
    p("Δρ измеряет различие половин поля. CV показывает неоднородность плотности по K блокам, RMS - среднеквадратичную скорость. Блоки имеют одинаковый вес.")
    table("Таблица 2. Параметры расчётов", [
        ["Параметр", "Значение"], ["Поле и длительность", "128 × 96 узлов; 800 шагов"],
        ["Начальное заполнение каждого канала", "Слева p=0.45, справа p=0.05"],
        ["Ожидаемые начальные плотности", "Слева 1.8, справа 0.2 частиц/узел"],
        ["Повторения", "12 состояний (seed 2026-2037) × 2 типа границ"],
        ["Интервал итогового усреднения", "Последние 100 шагов: t=701-800"]], [75,91])

    heading("3. Реализация и проверка")
    p("Программа написана на Python с NumPy и Matplotlib. Модель, расчёты и графики находятся в отдельных модулях. Файл main.py запускает всю лабораторную с готовыми параметрами и сохраняет результаты в папку results/.")
    p("Для каждого начального состояния выполняются оба варианта границ. Исходные массивы внутри пары одинаковы, поэтому сравнение не зависит от различий случайного заполнения. Для карт выбран seed=2026.")
    p("В цикле выполняются столкновение, подсчёт воздействия стенок, перенос и измерение. При нарушении баланса расчёт останавливается.")
    p("Листинг 1. Контроль импульса после шага", "caption")
    code('''post_collision = collide(state)
impulse += wall_impulse(post_collision, boundary)
state = stream(post_collision, boundary)
error = momentum(state) - p0 - impulse
if np.any(error):
    raise RuntimeError("Momentum balance failed")''')
    table("Таблица 3. Автоматические проверки: 30 тестовых случаев пройдены", [
        ["Проверка", "Ожидаемый результат"],
        ["Все 16 состояний узла", "Правило столкновения, сохранение массы и импульса"],
        ["Независимая поэлементная модель", "Полное совпадение массивов на 15 шагах для обоих типов границ"],
        ["Обратимость", "100 прямых и 100 обратных шагов возвращают исходный массив"],
        ["Одна частица", "Период W при замыкании и 2W при отражении"],
        ["Усреднение и начальные условия", "Правильная скорость, повторяемость, характеристики случайного заполнения и проверка ввода"]], [58,108])
    p("Итоговые показатели сначала усредняются за последние 100 шагов каждого расчёта, затем по 12 расчётам. После знака ± указано выборочное стандартное отклонение между начальными состояниями. Это показатель разброса, а не доверительный интервал.")
    p("Проверки подтверждают правила дискретной модели. Сопоставление с экспериментальными данными реального газа не выполнялось.")

    heading("4. Результаты")
    sub("4.1. Перенос между половинами поля")
    figure("mixing.png", "Рисунок 1. Разность плотностей, нормированная на начальную. Сверху - среднее d(t) и полоса ±1 SD; снизу - среднее |d(t)| по 12 расчётам.")
    p("Разность плотностей меняет знак: газ переносится из левой половины в правую и обратно. Амплитуда уменьшается, но колебания сохраняются до конца расчёта. Среднее |d(t)| помогает отличить выравнивание от взаимного сокращения положительных и отрицательных значений разных расчётов.")
    p("Положительные максимумы при периодических границах наблюдаются около t=182, 364, 543 и 720; при отражающих - около t=367 и 731. Период по поздним максимумам составляет примерно 180 и 364 шага. Эти оценки относятся к выбранной сетке и начальному заполнению.")
    p(f"Линия 0.1 показывает небольшое различие плотностей. Первые 25 последовательных шагов ниже этого порога начинаются в t={b['periodic']['threshold_time']} и t={b['reflecting']['threshold_time']}. Позже порог снова превышается, поэтому эти числа не являются временем установления равновесия.")

    story.append(PageBreak())
    sub("4.2. Поля плотности и скорости")
    figure("fields_periodic.png", "Рисунок 2. Периодические границы: плотность (сверху), скорость и направление потока (снизу). Seed=2026; t=0, 120, 800; блоки 8 × 8.", width=160)
    figure("fields_reflecting.png", "Рисунок 3. Отражающие границы с тем же начальным состоянием. Цветовые шкалы и масштаб стрелок совпадают с рисунком 2.", width=160)
    p("В поле остаются полосы неоднородной плотности и направленного потока. Периодические границы пропускают возмущение через край, а отражающие возвращают его внутрь. Это согласуется с различием периодов колебаний.")

    story.append(PageBreak())
    sub("4.3. Итоговые показатели")
    rows = [["Показатель за t=701-800", "Периодические", "Отражающие"]]
    for label,key in [("Среднее |Δρ|, частиц/узел", "tail_absolute_contrast"),
                      ("Неоднородность плотности (CV)", "tail_density_cv"),
                      ("RMS скорости, узлов/шаг", "tail_rms_speed")]:
        rows.append([label] + [fmt(b[bc][key+'_mean']) + " ± " + fmt(b[bc][key+'_sd']) for bc in ("periodic","reflecting")])
    rows += [["Максимальная ошибка массы", "0", "0"], ["Максимальная ошибка баланса импульса", "0", "0"]]
    table("Таблица 4. Среднее ± SD по 12 начальным состояниям", rows, [65,50.5,50.5])
    figure("observables.png", "Рисунок 4. Неоднородность плотности и RMS скорости для одного расчёта (seed=2026). Таблица 4 обобщает все 12 состояний.")
    p("В выбранном интервале периодические границы дают меньшую разность плотностей, меньшую неоднородность и меньшую скорость. Поскольку движение колебательное, вывод зависит от интервала наблюдения.")
    p("Число частиц и баланс импульса выполняются точно во всех 24 расчётах. Например, при отражении для seed=2026 изменение импульса к t=800 равно (3148, -50) и точно совпадает с накопленным импульсом стенок. Импульс самого газа при отражении может меняться.")

    heading("5. Выводы, запуск и источники")
    p("1. Реализована синхронная модель HPP с двумя типами границ. Выполнены 24 расчёта и пройдены 30 тестовых случаев. Масса и баланс импульса сохраняются на каждом шаге.")
    p("2. Границы меняют характер переноса газа. В последние 100 шагов средняя абсолютная разность плотностей составляет 0.287720 ± 0.005875 при периодических и 0.593613 ± 0.010993 при отражающих границах.")
    p("3. За 800 шагов сохраняются затухающие колебания. Устойчивое равновесие не подтверждено. Движение модели обратимо; уменьшение наблюдаемой амплитуды не доказывает необратимое затухание всей системы.")
    p("Исследование относится к одной сетке и одному типу начального распределения. Четыре направления движения создают анизотропию, но её отдельное измерение, изменение размеров поля и калибровка на реальный газ здесь не проводились.")
    sub("Как повторить работу")
    p("Требуется Python 3.10 или новее. В папке проекта выполняются две команды:")
    p("python -m pip install -r requirements.txt<br/>python main.py", "small")
    p("Готовые данные и четыре рисунка находятся в experiments/. Запуск сохраняет новые результаты в results/. Проверки: python -m pytest -q (после установки requirements-dev.txt). Сборка PDF: python report/build_report.py. Дополнительные параметры описаны в README.")
    p(f'Исходники и PDF: <link href="{REPO}" color="#246b9d">{REPO}</link>.', "small")
    sub("Источники")
    p("[1] «Лекция 4 - Решётчатый газ», разделы 3, 4.1, 5.1. Материалы курса, 12 с.", "small")
    p("[2] «О выполнении заданий по курсу», разделы 4-5. Материалы курса, 5 с.", "small")
    p("[3] «Лекция 3 - Клеточные автоматы», разделы 1-2. Материалы курса, 22 с.", "small")
    p('[4] Пример состава HPP-проекта: <link href="https://github.com/golichnicov/hpp_lattice_gas" color="#246b9d">github.com/golichnicov/hpp_lattice_gas</link>.', "small")
    p('[5] Пример титульного листа: <link href="https://github.com/alexanderchekhov8-wq/lab1.1" color="#246b9d">github.com/alexanderchekhov8-wq/lab1.1</link>, «Отчет по лабораторной работе 1.pdf».', "small")
    p("Реквизиты курса взяты из [5]; ФИО и группа заданы пользователем. Код и численные результаты других студентов не заимствованы.", "small")

    def footer(canvas, doc):
        if doc.page > 1:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#d6dfe6"))
            canvas.line(22*mm, 17*mm, 188*mm, 17*mm)
            canvas.setFont("Sans", 8)
            canvas.setFillColor(colors.HexColor("#617483"))
            canvas.drawString(22*mm, 12*mm, "HPP D2Q4 | Растегаев А. Э. | СМ6-91")
            canvas.drawRightString(188*mm, 12*mm, str(doc.page))
            canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=22*mm, leftMargin=22*mm,
                            topMargin=20*mm, bottomMargin=23*mm,
                            title=TITLE, author="Растегаев Алексей Эдуардович",
                            subject="Лабораторная работа по компьютерному моделированию")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    markdown = "\n".join(md).replace("<br/>", "\n")
    markdown = re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>', r'[\2](\1)', markdown)
    markdown = re.sub(r"(?m)^(\|[^\n]*\|)\n\n(?=\|)", r"\1\n", markdown)
    (ROOT / "report/report.md").write_text(markdown, encoding="utf-8")
    temporary.cleanup()
    print(f"Отчёт: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Собрать отчёт HPP из сохранённых результатов.")
    parser.add_argument("--output", type=Path, default=ROOT / "report/hpp_report.pdf")
    build(parser.parse_args().output)
