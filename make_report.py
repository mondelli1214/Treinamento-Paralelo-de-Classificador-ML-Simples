
import pandas as pd, os

TPL = open("report_template.md", "r", encoding="utf-8").read()

def fill_template(row):
    d = row.to_dict()
    txt = TPL
    for k, v in d.items():
        placeholder = "{{" + k + "}}"
        if placeholder in txt:
            txt = txt.replace(placeholder, str(v))
    # format numbers with extra placeholders
    for k in ["acc_train_seq","acc_test_seq","acc_train_par","acc_test_par",
              "train_time_seq","train_time_par","infer_time_seq","infer_time_par",
              "speedup_train","speedup_infer"]:
        ph = "{{" + k + ":.3f}}"
        ph4 = "{{" + k + ":.4f}}"
        if ph in txt:
            txt = txt.replace(ph, f"{float(row[k]):.3f}")
        if ph4 in txt:
            txt = txt.replace(ph4, f"{float(row[k]):.4f}")
    return txt

def main():
    csv_path = os.path.join("results", "results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("results/results.csv não encontrado. Rode `python benchmark.py` antes.")
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    report = fill_template(last)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Gerado report.md com base no último experimento.")

if __name__ == "__main__":
    main()
