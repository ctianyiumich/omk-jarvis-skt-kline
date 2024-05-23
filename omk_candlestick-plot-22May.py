import pandas as pd
import numpy as np
from datetime import datetime
import math

import rqdatac as rq
from dotenv import load_dotenv
import os
import json

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg

import matplotlib as mpl
import seaborn as sns

zh_font = "Songti SC"
mpl.rcParams['font.sans-serif'] = [zh_font]  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False
sns.set(font_scale=1.5, font=zh_font)

from warnings import filterwarnings

filterwarnings('ignore')

# Initiate Ricequant data
file_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv()
token_path = os.getenv("RQDATAC2_CONF")
os.environ['RQDATAC2_CONF'] = token_path
rq.init()
# All common stocks
CS_df = rq.all_instruments(type="CS")
# Create/Check existence of folders for json files and images
new_cache_path = f"{file_dir}/json_files"
new_img_path = f"{file_dir}/img_files"
if not os.path.exists(new_cache_path):
    os.makedirs(new_cache_path)
if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)


class IndividualStock:
    def __init__(self, ticker, start_date, end_date, frequency):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

    def load_cache(self):
        cache_json = f"{new_cache_path}/{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_prices.json"
        try:
            cache_file = open(cache_json, 'r')
            cache_file_contents = cache_file.read()
            cache = json.loads(cache_file_contents)
            cache_file.close()
        except FileNotFoundError:
            cache = False
        return cache

    def save_cache(self, cache):
        cache_json = f"{new_cache_path}/{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_prices.json"
        with open(cache_json, 'w', encoding='utf-8') as file_obj:
            json.dump(cache, file_obj, ensure_ascii=False, indent=2)

    def get_prices(self):
        return rq.get_price(
            self.ticker,
            self.start_date,
            self.end_date,
            self.frequency)

    def operate_cache(self):
        if self.load_cache():
            prices_json = self.load_cache()
            prices = pd.read_json(prices_json)
        else:
            prices = self.get_prices()
            prices_json = prices.to_json()
            self.save_cache(cache=prices_json)
        prices = prices[['open', 'close', 'high', 'low', 'volume']]
        prices_index = []
        for index in prices.index:
            try:
                new_index = str(index)
                new_index = new_index.split("p('")
                new_index_1 = list(new_index)[1]
                new_index = datetime.strptime(new_index_1.split(' ')[0], '%Y-%m-%d')
            except ValueError:
                new_index = None
            prices_index.append(new_index)
        prices.index = prices_index
        return prices

    def calculate_moving_averages(self, lags):
        moving_average_df = pd.DataFrame()
        prices = self.operate_cache()
        for lag in lags:
            moving_average_df[f'MA_{lag}'] = prices.close.rolling(lag).mean()
        return moving_average_df

    def separate_bull_bear(self):
        prices = self.operate_cache()
        bullish_sticks = prices[prices['open'] < prices['close']]
        bearish_sticks = prices[prices['open'] >= prices['close']]
        return bullish_sticks, bearish_sticks

    def plot_candlesticks(self, lags):

        bullish_sticks, bearish_sticks = self.separate_bull_bear()
        moving_average_df = self.calculate_moving_averages(lags)

        bar_width = 1
        whisk_width = 0.1 * bar_width
        ma_width = 5 * whisk_width

        green = 'green'
        red = 'red'
        facecolor = 'ghostwhite'
        plt.style.use('ggplot')

        canvas = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0.01, height_ratios=[4, 1])

        ax0 = canvas.add_subplot(spec[0])
        ax0.set_facecolor(facecolor)
        ax0.bar(bullish_sticks.index,
                bullish_sticks.close - bullish_sticks.open,
                bar_width,
                bottom=bullish_sticks.open,
                color=red)
        ax0.bar(bullish_sticks.index,
                bullish_sticks.high - bullish_sticks.close,
                whisk_width,
                bottom=bullish_sticks.close,
                color=red)
        ax0.bar(bullish_sticks.index,
                bullish_sticks.low -
                bullish_sticks.open, whisk_width,
                bottom=bullish_sticks.open,
                color=red)

        ax0.bar(bearish_sticks.index,
                bearish_sticks.close - bearish_sticks.open,
                bar_width,
                bottom=bearish_sticks.open,
                color=green)
        ax0.bar(bearish_sticks.index,
                bearish_sticks.high - bearish_sticks.open,
                whisk_width,
                bottom=bearish_sticks.open,
                color=green)
        ax0.bar(bearish_sticks.index,
                bearish_sticks.low - bearish_sticks.close,
                whisk_width,
                bottom=bearish_sticks.close,
                color=green)

        for lag in lags:
            ax0.plot(moving_average_df[f'MA_{lag}'], linewidth=ma_width, label=f'MA_{lag}')
        ax0.legend()

        ax0.tick_params(axis='x', colors='white')
        ax0.set_ylabel('Prices')

        ax1 = canvas.add_subplot(spec[1])
        ax1.set_facecolor(facecolor)
        ax1.bar(bullish_sticks.index, bullish_sticks.volume, color=red)
        ax1.bar(bearish_sticks.index, bearish_sticks.volume, color=green)
        ax1.set_ylabel('Volume')

        plt.xticks(rotation=30, ha='right')
        plt.suptitle(f'{get_name_by_code(code=self.ticker)}-{self.ticker}(frequency: {self.frequency})')

        save_file_path = f"{new_img_path}/{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_candle.png"
        canvas.savefig(save_file_path)

        plt.close()
        return save_file_path


def get_fig_dirs_list(xlsx_df, lag_terms_list):
    fig_dirs = []
    for i in range(len(xlsx_df)):
        stock = IndividualStock(
            ticker=xlsx_df.code[i],
            start_date=xlsx_df.start_date[i],
            end_date=xlsx_df.end_date[i],
            frequency=xlsx_df.frequency[i])
        fig = stock.plot_candlesticks(lags=lag_terms_list)
        fig_dirs.append(fig)
    return fig_dirs


def concat_plots(fig_dirs, lag_terms_list):
    # 我尝试的拼接方法只能拼一个完整矩阵的大图（小图数量是3的倍数），因此在引入超过2个个股时考虑引入dummy使总体小图数量刚好为6或9，最后再删去dummy以达到效果
    dummy_stock = IndividualStock(
        ticker='000001.XSHE',
        start_date='2020-01-01',
        end_date='2020-01-15',
        frequency='1d')
    num_queries = len(fig_dirs)
    if len(fig_dirs) > 1:
        nrow = math.ceil(len(fig_dirs) / 3)
        if nrow > 1:
            # 添加冗余的dummy_stock使list的长度为3的倍数
            if nrow * 3 - num_queries != 0:
                for k in range(nrow * 3 - num_queries):
                    fig_dirs.append(dummy_stock.plot_candlesticks(lags=lag_terms_list))
            fig, axes = plt.subplots(nrow, 3, figsize=(20, 20))
            for i, ax in enumerate(axes.flat):
                img = mpimg.imread(fig_dirs[i])
                ax.imshow(img)
                ax.axis('off')
            # 移除dummy_stock的图像
            if nrow * 3 - num_queries == 2:
                axes[nrow - 1, -1].set_visible(False)
                axes[nrow - 1, 1].set_visible(False)
            elif nrow * 3 - num_queries == 1:
                axes[nrow - 1, -1].set_visible(False)
        else:
            fig, axes = plt.subplots(1, num_queries, figsize=(20, 20))
            for i, ax in enumerate(axes.flat):
                img = mpimg.imread(fig_dirs[i])
                ax.imshow(img)
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{new_img_path}/concat_chart.png')


def get_name_by_code(code=str):
    try:
        return CS_df[CS_df.order_book_id == code].symbol.values[0]
    except ValueError:
        return None


def load_data(end_date):
    stk_watch_excel = pd.ExcelFile(f'{file_dir}/stk_watch_list.xlsx')
    stk_sheets = stk_watch_excel.sheet_names[2:-1]
    # 合并所有含有股票的sheets
    stk_watch_list = []
    for sheet in stk_sheets:
        stk_watch_pd = pd.read_excel(stk_watch_excel, sheet_name=sheet, header=None)
        stk_watch_list.append(stk_watch_pd)
    stk_watch_concat_pd = pd.concat(stk_watch_list)

    stk_watch_concat_pd.columns = stk_watch_concat_pd.iloc[0, :]
    stk_watch_concat_pd = stk_watch_concat_pd.drop(index=0)
    stk_watch_concat_pd = stk_watch_concat_pd.iloc[:, 2:]
    nrow = len(stk_watch_concat_pd)
    stk_watch_concat_pd.index = np.linspace(0, nrow - 1, nrow)
    stk_watch_concat_pd.index = [int(index) for index in stk_watch_concat_pd.index]
    stk_watch_concat_pd['股票代码'] = ['0' * (6 - len(str(code))) + str(code) for code in stk_watch_concat_pd.股票代码]
    stk_watch_concat_pd['股票代码'] = [code + '.XSHG' if code[0] == '6' else code + '.XSHE' for code in stk_watch_concat_pd.股票代码]

    result_df = pd.DataFrame({
        'code': stk_watch_concat_pd['股票代码'],
        'start_date': ['1900-01-01'] * len(stk_watch_concat_pd['股票代码']),
        'end_date': [end_date] * len(stk_watch_concat_pd['股票代码']),
        'frequency': ['1d'] * len(stk_watch_concat_pd['股票代码'])
    })
    return result_df


def main(xlsx_df, cancat=False):
    lag_terms_list = [5, 10, 20, 30, 60, 120, 250]
    fig_dirs = get_fig_dirs_list(xlsx_df, lag_terms_list)
    if cancat:
        concat_plots(fig_dirs, lag_terms_list)
    else:
        pass


if __name__ == '__main__':
    """
    xlsx_name = 'stocks_queries.xlsx'
    xlsx_df = pd.DataFrame(xlsx_towrite_dict)
    xlsx_df = pd.read_excel(f'{file_dir}/{xlsx_name}')
    """
    end_date_str = '2024-05-20'
    data_df = load_data(end_date_str)
    # All inputs given until this line
    main(data_df, cancat=False)
