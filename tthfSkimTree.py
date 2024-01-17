import ROOT

ROOT.EnableImplicitMT()

names = ROOT.std.vector("string")()
files = [
    "/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/Mu/RunResult_BvsC_Only_One_Step_20/Central_Syst/Vcb_TTLJ_powheg.root",
    "/data6/Users/isyoon/Vcb_Post_Analysis/Sample/2017/El/RunResult_BvsC_Only_One_Step_20/Central_Syst/Vcb_TTLJ_powheg.root",
]
for n in files:
    names.push_back(n)
df = ROOT.RDataFrame("Central/Result_Tree", names)
df = df.Define(
    "isBB",
    "for(size_t i = 0; i < Sel_Gen_HF_Origin.size(); ++i) if (Sel_Gen_HF_Flavour[i] == 5 && abs(Sel_Gen_HF_Origin[i]) != 6 && abs(Sel_Gen_HF_Origin[i]) != 24) return 1; return 0;",
)
df = df.Define(
    "isCC_temp",
    "for(size_t i = 0; i < Sel_Gen_HF_Origin.size(); ++i) if (Sel_Gen_HF_Flavour[i] == 4 && abs(Sel_Gen_HF_Origin[i]) != 6 && abs(Sel_Gen_HF_Origin[i]) != 24) return 1; return 0;",
)
# since both BB and CC cannot be 1 in same time
df = df.Define("isCC", "!isBB && isCC_temp")

df = df.Define(
    "whatMode",
    "(decay_mode == 21 || decay_mode == 23) ?2 : (decay_mode == 41 || decay_mode == 43) ? 4 : (decay_mode == 45) ? 45 : 0",
)

df = df.Filter("isBB || isCC")
columns = df.GetColumnNames()
columns_new = ROOT.std.vector("string")()
for i, element in enumerate(columns):
    if (
        element == "isBB"
        or element == "isCC"
        or element == "isCC_temp"
        or element == "whatMode"
    ):
        continue
    columns_new.push_back(element)

df.Snapshot("Central/Result_Tree", "Vcb_TTLJ_Powheg_TTHF.root", columns_new)
