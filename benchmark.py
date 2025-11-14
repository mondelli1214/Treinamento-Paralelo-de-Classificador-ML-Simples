
import argparse, time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mnist_io import load_mnist
from logreg import SoftmaxLogReg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=8, help="n_jobs para versão paralela e inferência")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    (X_train, y_train), (X_test, y_test) = load_mnist(args.data_dir, auto_download=not args.no_download)
    input_dim = X_train.shape[1]
    num_classes = int(y_train.max()) + 1

    os.makedirs("results", exist_ok=True)

    # Sequencial
    model_seq = SoftmaxLogReg(input_dim, num_classes, lr=args.lr)
    t0 = time.perf_counter()
    model_seq.fit_sequential(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    t1 = time.perf_counter()
    train_time_seq = t1 - t0
    t0 = time.perf_counter()
    acc_train_seq = model_seq.accuracy(X_train, y_train, jobs=1)
    acc_test_seq  = model_seq.accuracy(X_test, y_test, jobs=1)
    inf_time_seq  = time.perf_counter() - t0

    # Paralelo
    model_par = SoftmaxLogReg(input_dim, num_classes, lr=args.lr)
    t0 = time.perf_counter()
    model_par.fit_parallel(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, jobs=args.jobs, verbose=True)
    t1 = time.perf_counter()
    train_time_par = t1 - t0
    t0 = time.perf_counter()
    acc_train_par = model_par.accuracy(X_train, y_train, jobs=args.jobs)
    acc_test_par  = model_par.accuracy(X_test, y_test, jobs=args.jobs)
    inf_time_par  = time.perf_counter() - t0

    speedup_train = train_time_seq / train_time_par
    speedup_infer = inf_time_seq / inf_time_par

    row = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "jobs": args.jobs,
        "train_time_seq": train_time_seq,
        "train_time_par": train_time_par,
        "infer_time_seq": inf_time_seq,
        "infer_time_par": inf_time_par,
        "speedup_train": speedup_train,
        "speedup_infer": speedup_infer,
        "acc_train_seq": acc_train_seq,
        "acc_test_seq": acc_test_seq,
        "acc_train_par": acc_train_par,
        "acc_test_par": acc_test_par,
    }
    df = pd.DataFrame([row])
    out_csv = os.path.join("results", "results.csv")
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    # gráfico
    fig1 = plt.figure()
    plt.bar(["Treino Seq", "Treino Par"], [train_time_seq, train_time_par])
    plt.ylabel("Tempo (s)")
    plt.title("Tempo de Treino")
    fig1.savefig(os.path.join("results", "train_times.png"), bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.bar(["Inferência Seq", "Inferência Par"], [inf_time_seq, inf_time_par])
    plt.ylabel("Tempo (s)")
    plt.title("Tempo de Inferência")
    fig2.savefig(os.path.join("results", "infer_times.png"), bbox_inches="tight")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.bar(["Speedup Treino", "Speedup Inferência"], [speedup_train, speedup_infer])
    plt.ylabel("x (maior é melhor)")
    plt.title("Speedup")
    fig3.savefig(os.path.join("results", "speedups.png"), bbox_inches="tight")
    plt.close(fig3)

    print("OK! Resultados em results/ e results.csv")

if __name__ == "__main__":
    main()
