"""v2.1 pass-3 OA adjudication classifier (Medial/Lateral/PF OA).

Clause-level multilingual classifier over full radiology reports for the
pool-3 cells (v2 `uncertain` + public < 0.9). Per-clause polarity with
pre/post-pathology negation scope, compartment attribution where
patellar/facet/trochlear anatomy dominates medial-lateral adjectives
("medial patellar facet" is PF evidence, not Medial OA), severity-restricted
negations kept positive under the any-severity rule, hedged-only and
osteochondral/traumatic findings masked. Reads /tmp/pool3_oa.parquet
(rebuild from review_queue.parquet + studies.parquet); writes
/tmp/v21_pool3_oa_full.parquet. Post-run pins applied: study suffixes
56388411/34078389 PF OA -> positive, 76106854/99606924 Medial OA -> positive.
Run: PYTHONPATH=. python scripts/research/v21_pool3_oa_final.py
"""

import re

import pandas as pd

df = pd.read_parquet("/tmp/pool3_oa.parquet")
s = pd.read_parquet("initiative/releases/report-supervision-v1/studies.parquet")[
    ["study_uid", "Report"]
]
df = df.merge(s, on="study_uid", how="left")

PATH = re.compile(
    r"hondromalaci|chondromala|chondropath|chondros|condropat|osteoarthr|gonarthr|osteofit|osteophyt|"
    r"ostéophyt|\bartros\w*|\barthros\w*|\barthrose\w*|\barthrit\w*|degenerativ|dejenerat|degeneer|degenere|dejenerasy|"
    r"thinn\w*|ausgedünn|ausgedeünn|verdunn|reducirana|stanjen|изтънен|изтъняване|aminciss|"
    r"atténuat|fissur|fisur|narrow|daral|sužen|wearing|wear\b|denud|spur|épine|sivrileş|fraying|"
    r"subchondral|kemik iliği|koštani? edem|marrow edema|botoedeem|Knochenmarksödem|kemik iliği ödemi|"
    r"joint space|incelme|bütünlük kaybı|geode|субхондрал|gonartroz|остеофит|εκφύλιση|επιχείλι|"
    r"οστεοφύτ|οστεοαρθρίτιδ|χονδροπάθεια|διάβρωση|λέπτυνση|úlceras|úlcera|ulcer|osteoartritičke|"
    r"gonartrotske|artritičke|osteochondral|osteocondral|osteohondral|cartilagenous|cartilagineo|"
    r"cartilagineous|defect|defekt|дефект|костномозъчен|kayb|izražene|ulceras|denuded|"
    r"cartilage loss|bütünlük|degeneration|dejenerasyon|dejenerasyonu|osteocartilagenous|"
    r"osteocartilagine|\bOA\b|kraakbeenlijden|verdund|lijden graad|ausdünn|dégénérat|"
    r"perte de cartilage|degenerative|arthrosique|artrózis|osteoarthrosis|chondropathie|"
    r"irregularit|iregularit|nepraviln|лизиран|улцериран|улцера|лезия на хрущял|diabrosi|"
    r"chondral loss|kraakbeenletsel|letsels|wijzigingen|veranderingen",
    re.I,
)
MED = re.compile(
    r"\bmedial\w*|anteromedia\w*|posteromedia\w*|medijaln\w*|interne|innen\b|"
    r"internal\w*|médial|antéromédial|έσω|медиалн\w*|iç\s|interno|interna|unutrašnj",
    re.I,
)
LAT = re.compile(
    r"\blateral\w*|anterolatera\w*|posterolatera\w*|lateraal|externe|außen|"
    r"external\w*|latéral|antérolatéral|postérolatéral|έξω|латералн\w*|ekstern|"
    r"dış\s|externo|externa|spoljašnj",
    re.I,
)
PFP = re.compile(
    r"patellofemoral|patelofemoral|patel|патела|trochl|trócle|tróchle|retropatellar|"
    r"femoropatel|fémoropatel|επιγονατ|rotul|rótula|troklear|trohlearn|Patellafacette|"
    r"Patellarückfläche|\bFP\b",
    re.I,
)
FTW = re.compile(
    r"femorotibial|fémorotibial|femorotibijal|tibiofemoral|tibio-femoral|\bFT\b|gonartroz|"
    r"gonarthr|osteoarthr|arthrose|artrose|arthrosis|koljen|knee joint|diz eklemi|zgloba|zoglob|"
    r"koljena|koljeno|rodilla|articular cartilage|intra-articular cartilage|artikulacijsk\w+ "
    r"hrskavic|zglobni\w+ hrskavic|femoralne i tibijalne|tibijalne i kondilarne|kon(?:dila|dil)|"
    r"condyle|kondil|kondila|kondyly|condyles|condila|condylar|kondilarn|condilar|tibial|tibijaln|"
    r"tibialne|plato|platou|plateau|femoral|femoraln|femur|femora|tibia|tibialn|колянната става|"
    r"stavnият|става|γόνατ|joint|articular|intra-articular|zglob|eklem|articulación|"
    r"mesafes|aralığı",
    re.I,
)
NONOA = re.compile(
    r"menisc|menisk|µηνίσκ|menisko|ligament|bağ|cruciat|križnog|kreuzbänder|kollateral|collateral|"
    r"лигамент|връзк|tendon|tetiv|plicae|plica|bursa\b|ganglion|retinacul|parameniscal|paramenisc|"
    r"popliteal|Hoffa|quadriceps|patellar tendon|it band|iliotibial|effusion|derrame|izliv|излив|"
    r"épanchement|vocht|sıvı|hydrops|bursitis|tendinosis|tendinitis|sprain|distorsi|esguince|"
    r"subcutaneous|prepatellar|soft tissue|mekih tkiva|nervo|vessel|neurovascular|intercondylar|"
    r"intercondílea|notch|intercondyloid|χιαστ|σύνδεσµ|свръзк|връзки|desma|retinakül",
    re.I,
)
CART = re.compile(
    r"cartilag|kıkırdak|hrskavic|хрущял|χόνδρ|kraakbeen|Knorpel|hondro|chondr|condro|"
    r"cartílago|joint space|osteofit|osteophyt|ostéophyt|trnovi|sivrileş|spur|épine|"
    r"subchondral|επιχείλι|οστεοφύτ|gonarthr|gonartroz|osteoarthr|artritičke|gonartrotske|"
    r"artros|arthros|denud|хрущяла|osteochondral|osteocondral|osteohondral|"
    r"cartilagenous|cartilagineo|cartilagineous|osteocartilagenous|sužen|narrow|daral",
    re.I,
)
ALLC = re.compile(
    r"three compartments|all three|tricompartment|trikompartm|trois compartiments|sva tri|"
    r"tricompartiment|both joint|αµφότερα|obe\b|both compartments|αµφοτερόπλευρα",
    re.I,
)
ALLC_ML = re.compile(
    r"medial and lateral|medialnih i lateralnih|mediaal en lateraal|"
    r"interne et externe|medialnog i lateralnog|medialen und lateralen|"
    r"médial et latéral",
    re.I,
)
NEGWORD = re.compile(
    r"\bno\b|\bwithout\b|\bnone\b|\babsen[ct]|\bintact\w*|\bnormal\w*|ureda?n\w*|primjeren|behouden|"
    r"preserved|preservada?o?s?|maintained|sağlam|normaldir|\bb\.o\.\b|regelrecht|unauffällig|\bhomogen\w*|"
    r"запазен|неизтънен|нормалн|\bbez\b|\bkein\w*|\bgeen\b|\bnema\b|\bnije\b|ne nalazi|ne vidi|\bsin\b|"
    r"\bpas\b|χωρίς|δεν παρατηρείται|απουσία|\bingen\b|\byoktur\b|\byok\b|saptanmadı|izlenmedi|"
    r"gözlenmedi|\bfri\b|free of|očuvan|σαφής|ικανοποιητικ|zachowan|nicht verdickt|no hay|"
    r"no se (?:evidencia|observa)|zonder|\bohne\b|senza\b|fără|\bnincs\b|\bбез\b|no evident|"
    r"no alteration|no focal|well-maintained|well maintained|\bintakte|unverändert|physiologisch|"
    r"physiologique|physiologično|физиолог|normální|\bnormale\b|\bnormalni\b|\bnormalna\b|"
    r"\bordinary\b|\bunremarkable\b|\bnegative for\b|\bnot observed\b|\bnot seen\b|\bnot present\b|"
    r"\babsent\b",
    re.I,
)
NEG_AFTER_FIND = re.compile(
    r"not (?:seen|identified|detected|observed|noted|demonstrated|present|appreciated|shown|found)|"
    r"yoktur|saptanmadı|saptanmamıştır|izlenmedi|izlenmemiştir|gözlenmedi|görülmedi|\bnije\b|\bnema\b|"
    r"нема|δεν παρατηρ|nicht (?:vorlieg|nachweis|darstellbar)|is absent|are absent|not present|"
    r"not demonstrated|not identified|not seen|not observed|not detected|kalmamış|no longer|"
    r"is not (?:seen|identified|detected|observed|noted|present)|are not (?:seen|identified)|"
    r"not observed|absence of|absent\b|intrameniskal",
    re.I,
)
SEV_NEG = re.compile(
    r"(?:\bno\b|kein\w*|geen\b|ohne\b|bez\b|nema\b|pas de|sans|δεν|χωρίς|nicht\b|\bsin\b|"
    r"yok\b|nije\b|zonder|without|there (?:is|are) no|il n.y a pas|no hay)\s+"
    r"(?:[^,.;]{0,30}?)(?:focal|fokal|deep|tief|greif|höhergrad|full[ -]?thickness|significant|"
    r"signifikant|wesentlich|majeure|manifest|izražen\w*|vidljiv|evident|relevante|relevant|"
    r"high-grade|higher-grade|grade 3|grade 4|substantial|meaningful|major|dute|complet|"
    r"complete|svid|expansive|extensive|extensif)",
    re.I,
)
HEDGE = re.compile(
    r"cannot be excluded|nicht ausgeschlossen|ne može se isključ|no se puede descartar|"
    r"peut pas être exclu|não pode ser excluída|možno vylúčiť|non si può escludere|"
    r"nie można wykluczyć|niet uitgesloten|nicht sicher auszuschließen|kakva?nijemi|"
    r"eventuell|eventual|möglicherweise|possible\b|moguće|muhtemel|vermutlich|éventuel|"
    r"suspected|verdacht|sospech|osumnjičen|\bR/O\b|rule out|r\.o\.",
    re.I,
)
CLAUSE_SPLIT = re.compile(
    r"[,;]|\bbut\b|\bjedoch\b|\bdoch\b|\bmaar\b|\bali\b|\bhowever\b|\balthough\b|\bwenngleich\b|"
    r"\bexcept\b|\bdışında\b|\bosim\b|\bwährend\b|\bwhile\b|\bwhereas\b|\byet\b",
    re.I,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])(?=[A-ZÇĞİÖŞÜŽŠĐĆČΑ-ΩА-Я])|(?<=[.!?])\s+|\n+")


def clause_polarity(cl):
    m_path = PATH.search(cl)
    m_neg = NEGWORD.search(cl)
    has_cart = bool(CART.search(cl))
    if not (m_path or has_cart):
        return None, has_cart
    if m_path and not m_neg:
        return "pos", has_cart
    if m_neg and not m_path:
        return "neg", has_cart
    if not m_path or not m_neg:
        return "neg" if m_neg else None, has_cart
    if m_neg.start() < m_path.start():
        return "neg", has_cart
    if NEG_AFTER_FIND.search(cl):
        return "neg", has_cart
    return "pos", has_cart


FT_BONE = re.compile(
    r"femorotibial|fémorotibial|femorotibijal|tibiofemoral|tibio-femoral|condyle|"
    r"condyles|kondil|kondila|condilar|kondila|platou|plato\b|plateau|tibia\b|tibial|"
    r"tibijaln|tibialn|femur\b|femoral cond|joint space|mesafes|menisk|menisc|"
    r"µηνίσκ|collateral|ligament|cruciat|compartment|kompartment|компарт|mesá|diaμέ|bölme|cebe|odjeljak",
    re.I,
)
PATEL_SITE = re.compile(
    r"trochl|trócle|tróchle|patel|патела|rotul|rótula|faset|facet|Facette|"
    r"retropatellar|Patellafacette|Patellarückfläche",
    re.I,
)


def clause_comp(cl):
    pff = bool(PFP.search(cl))
    med = bool(MED.search(cl))
    lat = bool(LAT.search(cl))
    ft = bool(FTW.search(cl))
    if ALLC.search(cl):
        return {"m", "l", "p"}
    # patellar/trochlear/facet anatomy: medial/lateral adjectives modify patellar facets, not FT compartments
    if pff and PATEL_SITE.search(cl) and not FT_BONE.search(cl):
        return {"p"}
    if ALLC_ML.search(cl) and (ft or FT_BONE.search(cl)):
        return {"m", "l", "p"}
    comps = set()
    if med:
        comps.add("m")
    if lat:
        comps.add("l")
    if pff:
        comps.add("p")
    if ft and not (med or lat):
        comps.add("j")
    return comps


def classify(target, rep, ev):
    sents = [x.strip() for x in SENT_SPLIT.split(rep or "") if x and x.strip()]
    if ev:
        sents.append(str(ev))
    need = {"Medial OA": "m", "Lateral OA": "l", "PF OA": "p"}[target]
    pos_hits = set()
    neg_hits = set()
    pos_ev = []
    neg_ev = []
    for x in sents:
        prev_pol = None
        prev_comps = None
        for cl in CLAUSE_SPLIT.split(x):
            cl = cl.strip()
            if len(cl) < 3:
                continue
            if re.fullmatch(
                r"(?i)(?:joint space|patellofemoral(?: compartment)?|cartilage|bones?|soft tissue|"
                r"menisci?|ligaments?|findings?|conclusion|impression|mri|knee|muscle|extensor "
                r"mechanism|synovial|fluid|effusion|bursae?|fractures?|alignment|hallazgos|befund|"
                r"beurteilung|sonstiges)\s*:?",
                cl,
            ):
                continue
            if HEDGE.search(cl):
                continue
            sev = SEV_NEG.search(cl)
            if sev:
                pm = PATH.search(cl)
                if pm and pm.start() < sev.start():
                    pass  # positive assertion precedes the severity-restricted negation
                else:
                    continue
            pol, has_cart = clause_polarity(cl)
            if pol is None:
                if has_cart and prev_pol and clause_comp(cl):
                    pol = prev_pol
                else:
                    continue
            else:
                prev_pol = pol
            if (
                NONOA.search(cl)
                and not has_cart
                and not re.search(
                    r"osteofit|osteophyt|ostéophyt|gonarthr|gonartroz|osteoarthr|artritičke|"
                    r"joint space|sužen|narrow|daral|subchondral",
                    cl,
                    re.I,
                )
            ):
                continue
            # a clause whose only pathology is an osteochondral lesion (traumatic, not OA) stays masked
            if (
                pol == "pos"
                and re.fullmatch(
                    r"(?i)[^.]{0,120}?(?:osteochondral|osteocondral|osteohondral|osteokondr|"
                    r"osteocartilag|osteochondrite?)[^.]{0,120}?",
                    cl,
                )
                and not re.search(
                    r"osteofit|osteophyt|ostéophyt|chondromalaci|chondropath|hondromalaci|"
                    r"thinn|denud|narrow|sužen|daral|degenerativ|dejenera|arthros|artros|"
                    r"gonarthr|gonartroz|subchondral",
                    re.sub(
                        r"osteochondral|osteocondral|osteohondral|osteokondr|osteocartilag", "", cl
                    ),
                    re.I,
                )
            ):
                continue
            comps = clause_comp(cl)
            if pol == "pos":
                if not comps and prev_comps:
                    comps = prev_comps
                pos_hits |= comps if comps else {"j"}
                pos_ev.append(cl)
                prev_comps = comps if comps else prev_comps
            else:
                if comps:
                    neg_hits |= comps
                elif has_cart:
                    neg_hits |= {"j"}
                neg_ev.append(cl)
    if need in pos_hits:
        return "positive", pos_ev[0][:200]
    if need in neg_hits:
        return "negative", neg_ev[0][:200]
    if "j" in pos_hits:
        if target == "PF OA":
            return "uncertain", "joint-level positive only"
        return "positive", pos_ev[0][:200] if pos_ev else "joint-level"
    if "j" in neg_hits:
        return "negative", neg_ev[0][:200] if neg_ev else ""
    if pos_hits:
        return "uncertain", "positive in other compartment"
    if neg_hits:
        return "uncertain", "negation in other compartment"
    return "uncertain", "no named statement"


calls = []
for r in df.itertuples():
    call, why = classify(r.target, r.Report, r.v2_evidence)
    calls.append(
        dict(
            study_uid=r.study_uid,
            target=r.target,
            public_value=r.public_value,
            call=call,
            why=why[:200],
        )
    )
out = pd.DataFrame(calls)
print(out.groupby(["target", "call"]).size())
out.to_parquet("/tmp/v21_pool3_oa_full.parquet", index=False)
print()
print("--- pub>=0.3 uncertain ---")
hi = out[out.public_value >= 0.3]
for r in hi[hi.call == "uncertain"].itertuples():
    print(f"{r.target[:9]:9} {r.study_uid[-8:]} pub={r.public_value:.2f} | {r.why[:90]}")
print()
print("--- pub>=0.3 negative ---")
for r in hi[hi.call == "negative"].itertuples():
    print(f"{r.target[:9]:9} {r.study_uid[-8:]} pub={r.public_value:.2f} | {r.why[:100]}")
