"""v2.1 pass-3 fluid adjudication classifier (Effusion, Baker's cyst).

Clause-level multilingual classifier over full radiology reports for the
484 pool-3 fluid cells (v2 `uncertain` + public < 0.9). Severity-qualified
and bare effusion/Baker's positives under the any-severity rule; bursal/
recess fluid excluded for effusion but semimembranosus-gastrocnemius bursa
fluid counts for Baker's. Reads review_queue.parquet + studies.parquet;
writes /tmp/v21_pool3_fluid.parquet. Manual pins in EFF_PIN/BAK_PIN.
Run: PYTHONPATH=. python scripts/research/v21_pool3_fluid.py
"""

import re

import pandas as pd

s = pd.read_parquet("initiative/releases/report-supervision-v1/studies.parquet")[
    ["study_uid", "Report"]
]
q = pd.read_parquet(
    "initiative/experiments/competitive-v1/label-source-ablation-v1/review_queue.parquet"
)
pool = q[
    (q.target.isin(["Effusion", "Baker's"])) & (q.v2_state == "uncertain") & (q.public_value < 0.9)
].merge(s, on="study_uid")

EFF_POS = re.compile(
    r"(?:trace|minimal|mild|small|moderate|large|marked|kleine|geringe|leve|mínimo|petit|léger|bescheiden|matig|beperkt|minimális|минимален|малък|умерен|малко|mali|manji|umjeren|minimaal|minimalni)[^.]{0,40}(?:effusion|derrame|épanchement|hydrops|Erguss|izliv|излив|effusa|versamento|vocht|effüzyon|efüzyon)|"
    r"(?:effusion|derrame|épanchement|hydrops|Erguss|ύδραρθρ|izliv|излив|effusa|versamento|vocht|effüzyon|efüzyon|Derrame)[^.]{0,50}(?:trace|minimal|mild|small|moderate|large|klein|gering|leve|mínimo|petit|matig|малък|mali|sinovitis|subcutaneous)|"
    r"fluid (?:accumulation|collection)[^.]{0,50}(?:suprapatellar|joint|articular|recess)|"
    r"(?:suprapatellar bursa|Suprapatellar bursada|bursada|lateral recess|patellofemoral eklem)[^.]{0,35}(?:sıvı artışı|effüzyon|efüzyon)|"
    r"sıvı artışı izlenm|sıvı miktarı[^.]{0,15}artmış|vrij vocht|fluid-filled Baker|vocht in de fossa poplitea|"
    r"ставен излив|eklem sıvı|tekućeg sadržaja|mayii|hydarthros|exsudát|"
    r"Μικρή[^.]{0,10}ενδαρθρική συλλογή υγρού|Συλλογή υγρού παρατηρείται ενδαρθρικά|"
    r"Minime épanchement intra-articulaire(?!.{0,60}physiologique)|"
    r"minimal efüzyon|minimal effüzyon|hafif[^.]{0,10}ef[üf]zyon|Eklem mesafeisnde[^.]{0,60}efüzyon",
    re.I,
)
EFF_NEG = re.compile(
    r"no (?:significant |obvious |relevant |clinically |evidence of (?:a )?(?:significant )?|increased |knee |joint |manifest |significant amount of )*(?:knee |joint |articular |intra-articular |degenerative )*(?:effusion|derrame|épanchement|hydrops|Erguss|излив|izliv|effusa|versamento|vocht|fluid)|"
    r"No (?:significant |obvious )?(?:amount of )?fluid within knee joint|"
    r"(?:No hay|sin|significativo|evidente)[^.]{0,25}derrame|signos que sugieran[^.]{0,25}derrame|No (?:se evidencia|hay)[^.]{0,30}derrame|"
    r"pas d.épanchement|Pas de liquide|keine?r? (?:wesentliche?r?|relevanter|größerer|manifester)? ?(?:Gelenk|Kniegelenk|knie)? ?[Ee]rguss|ohne relevanten Gelenkerguss|"
    r"geen (?:significant?e? |manifeste )?(?:hydrops|effusie|vocht|opzetting)|без излив|няма (?:МР )?данни за ставен излив|няма излив|"
    r"bez izliva|nema (?:značajnijeg |značajnog )?izliva|sıvı artışı saptanmamıştır|absence of effusion|no se evidencia derrame|ingen effusion|bez výpotku|no hydrops|"
    r"Χωρίς[^.]{0,45}υγρού|∆εν παρατηρείται[^.]{0,45}υγρού|Απουσία[^.]{0,45}υγρού|"
    r"Joint effusion\s*[:;]\s*None|Joint effusions\s*[:;]\s*None|"
    r"no significant[^.]{0,30}(?:effusion|Baker)|Nema značajnog intraartikularnog izljeva|"
    r"no evidence of significant Baker",
    re.I,
)
BAK_POS = re.compile(
    r"baker'?s? (?:cyst|kist|kisti|zyste|quiste|ciste|kistler?)|popliteal (?:cyst|cist|kist|zyst)|bekerova kista|Бекеров|бейкер|"
    r"fluid (?:collection|accumulation)[^.]{0,80}popliteal|popliteal fossa[^.]{0,80}(?:fluid|cyst)|"
    r"bursa (?:semimembranos|gastrocnem)[^.]{0,40}(?:vocht|fluid)|vocht[^.]{0,40}bursa (?:semimembranos|gastrocnem)|"
    r"baker kisti|Baker-Zyste|Baker-?cyste|poplitealna cista|poplitealnu cistu|Baker cisto|bursitis[^.]{0,40}gastrocnem|"
    r"Kyste de Baker\s*[:;]\s*(?!Aucun)(?:petit|minuscule|kyste|collabé)|kyste de Baker (?:minuscule|collabé|petit)|"
    r"poplitealne ciste|poplitealnoj burzi|Baker burzi|poplitealnoj bursi|Bakerovoj burzi|"
    r"Gastroknemi\w*[- ]?semimembrano\w* bursasındaki sıvı|gastrocnemius[^.]{0,10}semimembranosus[^.]{0,20}bursa|"
    r"Početno formiranje poplitealne|Quiste sinovial posterior|synovial cyst|Baker-Position|"
    r"subcentimeter cysts at the popliteal fossa|popliteal bursitis|popliteal düzeyde[^.]{0,20}effüzyon|"
    r"kas planları arasında[^.]{0,20}sıvı|synovial-lined lesion[^.]{0,80}popliteal|"
    r"κύστη Baker|manje poplitealne ciste|Sitna poplitealna cista|Poplitealna cista|popliteal fossa[^.]{0,60}cyst|"
    r"izljeva u burzi m\. ?semimembran|izljeva u Bakerovoj",
    re.I,
)
BAK_NEG = re.compile(
    r"no (?:significant )?(?:popliteal|Baker'?s?) (?:cyst|cysts)|ni quistes popl|quistes poplíteos|"
    r"pas de kyste (?:poplité|de Baker)|Kyste de Baker\s*[:;]\s*Aucun|kyste de Baker significatif|keine Baker-?Zyste|geen baker|"
    r"Ne nalazi se[^.]{0,30}(?:poplitealne|Bakerove) ciste|Bez (?:znakova )?poplitealne ciste|Bez znakova poplitealne|"
    r"bez formirane Baker ciste|bez formirane ciste|"
    r"без Бекерова|no popliteal cysts|baker kisti (?:yok|izlenmedi)|no Baker|"
    r"Popliteal fossada belirgin yer işgal eden lezyon saptanmadı|"
    r"No hay quiste popl|No se observa quiste de Baker|"
    r"nema značajnog[^.]{0,20}izljeva u Bakerovoj|no evidence of significant Baker",
    re.I,
)

EFF_OVERRIDE = {}  # study suffix -> call, set below


def decide(rep, target):
    sents = [x.strip() for x in re.split(r"(?<=[.!?:;\n])\s+|\n+", rep or "") if x.strip()]
    sents += [
        m.group(0)
        for m in re.finditer(
            r"(?:Épanchement articulaire|Joint effusion|Kyste de Baker)\s*[:;]\s*[^.:;\n]{0,120}",
            rep or "",
            re.I,
        )
    ]
    pos_s = []
    neg_s = []
    if target == "Effusion":
        for x in sents:
            if EFF_NEG.search(x):
                neg_s.append(x)
            elif EFF_POS.search(x):
                pos_s.append(x)
    else:
        for x in sents:
            if BAK_NEG.search(x):
                neg_s.append(x)
            elif BAK_POS.search(x):
                pos_s.append(x)
    if pos_s:
        return "positive", pos_s[0]
    if neg_s:
        return "negative", neg_s[0]
    return "uncertain", ""


# cells verified individually against full report (suffix -> call)
EFF_PIN = {
    "98904411": "uncertain",  # accentuated fluid, no Erguss named
    "61923815": "uncertain",
    "99606924": "negative",  # "ohne relevanten Gelenkerguss"
    "68443620": "uncertain",  # "keine größere Ergussbildung" hedged
    "77772269": "positive",  # bilateral: one knee "Μικρή ενδαρθρική συλλογή υγρού"
    "11182816": "uncertain",  # recess fluid + explicit "no joint fluid" — contradictory
    "15208375": "uncertain",  # same contradiction
    "93367497": "uncertain",  # extraarticular ganglion only
    "66956410": "uncertain",  # pes anserinus bursal fluid
    "79097099": "uncertain",
    "51538716": "uncertain",
    "99448897": "uncertain",  # trace Baker's only
    "58420811": "positive",  # "Minime épanchement intra-articulaire"
    "25722483": "uncertain",  # "potentiellement d'ordre physiologique"
    "49003589": "uncertain",  # "d'allure physiologique"
    "06773871": "negative",  # prepatellar fluid + "no fluid within knee joint"
    "91644995": "uncertain",  # Baker-bursa fluid is not joint effusion
    "78189902": "uncertain",
    "35174299": "uncertain",
    "09134762": "uncertain",
    "02173508": "uncertain",
    "93828847": "uncertain",
    "42012883": "uncertain",
    "96096520": "uncertain",
    "91426727": "uncertain",
}
BAK_PIN = {
    "91644995": "uncertain",  # "bez formirane ciste" — fluid but explicitly no formed cyst
    "78189902": "positive",  # "Manje izljeva u poplitealnoj burzi"
    "35174299": "positive",  # "manja količina izljeva u burzi m. semimembranosus"
    "09134762": "positive",  # "vidljivu manju poplitealnu cistu promjera 11 mm"
    "18051246": "positive",  # "Poplitealna cista promjera 4 mm"
    "74122989": "positive",  # "Sitna poplitealna cista promjera 4 mm"
    "23478921": "positive",  # "Poplitealna cista 10x4 mm"
    "74297358": "positive",  # "Baker cista dimenzija 23x12x49 mm"
    "96228004": "positive",  # "doch evenwel een Baker-cyste"
    "07098172": "positive",  # "Početno formiranje poplitealne ciste"
    "90302635": "positive",
    "59839029": "positive",  # "Quiste sinovial posterior"
    "27877590": "positive",  # "auch in Baker-Position"
    "05182876": "positive",  # "subcentimeter cysts at the popliteal fossa"
    "70466645": "positive",  # "µικρή συνοβιακή κύστη Baker"
    "42240776": "positive",  # "popliteal bursitis"
    "91667447": "positive",  # "Popliteal fossada kas planları arasında effüzyon ... sıvı"
    "48931799": "positive",  # "atypical Baker's cyst ... 3 cm"
    "00526112": "positive",  # "probably synovial cyst" posterior medial capsule
    "31248835": "positive",  # "Baker-cyste ... multiloculaire"
    "80168192": "positive",  # semimembranosus-gastrocnemius bursa fluid loculation
    "73383874": "positive",
    "41987775": "positive",  # "Pequeño quiste poplíteo"
    "99448897": "positive",  # "Trace Baker's cyst"
    "90218293": "uncertain",  # only effusion named; cysts silent
    "94277092": "uncertain",
    "43879156": "uncertain",
    "06023084": "uncertain",  # popliteus-tendon bursa fluid
    "96096520": "uncertain",  # parameniscal/ganglion cysts
    "21946816": "uncertain",  # parameniscal cyst
    "16463426": "negative",  # "Keine größere Baker-Zyste"
    "85112825": "uncertain",  # "moguća manja poplitealne ciste" — possible only
    "25722483": "negative",  # "sans kyste de Baker significatif"
    "49003589": "negative",
    "58420811": "negative",
    "71058799": "negative",  # "yer işgal eden lezyon saptanmadı" in popliteal fossa
    "67037309": "uncertain",
}

cnt = {}
out = []
for r in pool.itertuples():
    call, ev = decide(r.Report, r.target)
    suf = r.study_uid[-8:]
    pin = EFF_PIN.get(suf) if r.target == "Effusion" else BAK_PIN.get(suf)
    if pin:
        call = pin
    cnt[(r.target, call)] = cnt.get((r.target, call), 0) + 1
    out.append(
        dict(
            study_uid=r.study_uid,
            target=r.target,
            v2_state="uncertain",
            public_value=r.public_value,
            v21_call=call,
            evidence=ev[:300],
        )
    )
res = pd.DataFrame(out)
print(cnt)
res.to_parquet("/tmp/v21_pool3_fluid.parquet", index=False)
