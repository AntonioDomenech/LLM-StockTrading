import io
import matplotlib.pyplot as plt

def plot_price_and_equity(dates, prices, equity):
    fig, ax = plt.subplots()
    ax.plot(dates, prices, label="Price")
    ax.plot(dates, equity, label="Equity")
    ax.legend()
    ax.set_title("Price vs Equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
