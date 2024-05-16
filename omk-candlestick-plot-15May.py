import pandas as pd
import numpy as np
from datetime import datetime
import math

import rqdatac as rq
import os
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg

file_dir = os.path.dirname(os.path.realpath(__file__))

os.environ['RQDATAC2_CONF'] = 'rqdatac://license:YzvfmTPOTbdAxp_MF8ZkUnz731R0DrtRIxVsilKqhCAYQ7-l9gcbYca_AuOfLC92r5klaipygctYOZ3L__bb8VgojvyODdyu3mzukAggRjmBRsLyX4evVF1GuH1g6P5sW7WLrnsMof4efwOLJpCd0-LtJQOCBW2_6KkJ72FCScE=h5dtXsJHG7Yodqmtktbt_rfiZAHosK_3XB0UhcL_DM7ee-bmm5V_cftBhm0_gsynWTCAOwFZtHqzuSIkr8CzOyUoa9ucsD8PUw4VbMc2SrhJekRsvFmV6R5TtqHdZdHFpMti6KWIwKYhrZG9NRz6qUjx28P4Gs8hapB99FlHGmk=@rqdatad-pro.ricequant.com:16011'
rq.init()

class IndividualStock:
    def __init__(self, ticker, start_date, end_date, frequency):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

    def load_cache(self):
        CACHE_json = f"{os.path.dirname(os.path.realpath(__file__))}/{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_prices.json"
        try:
            cache_file = open(CACHE_json, 'r')
            cache_file_contents = cache_file.read()
            cache = json.loads(cache_file_contents)
            cache_file.close()
        except:
            cache = False
        return cache

    def save_cache(self, cache):
        CACHE_json = f"{file_dir}/{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_prices.json"
        with open(CACHE_json, 'w', encoding='utf-8') as file_obj:
            json.dump(cache, file_obj, ensure_ascii=False, indent=2)
    
    def get_prices(self):
        return rq.get_price(self.ticker, self.start_date, self.end_date, self.frequency)

    def operate_cache(self):
        if self.load_cache():
            prices_json = self.load_cache()
            prices = pd.read_json(prices_json)
        else:
            prices = self.get_prices()
            prices_json = prices.to_json()
            self.save_cache(cache=prices_json)
        prices = prices[['open','close','high','low','volume']]
        prices.index = [datetime.strptime(str(index).split("p('")[1].split(' ')[0],'%Y-%m-%d') for index in prices.index]
        return prices

    def calculate_MAs(self,lags):
        MA_df = pd.DataFrame()
        prices = self.operate_cache()
        for lag in lags:
            MA_df[f'MA_{lag}'] = prices.close.rolling(lag).mean()
        return MA_df

    def separate_bull_bear(self):
        prices = self.operate_cache()
        bullishSticks = prices[prices['open']<prices['close']]
        bearishSticks = prices[prices['open']>=prices['close']]
        return bullishSticks, bearishSticks
    
    def plot_candlesticks(self,lags=[5, 10, 20, 30, 60, 120, 250]):

        bullishSticks, bearishSticks = self.separate_bull_bear()
        MA_df = self.calculate_MAs(lags)

        bar_width = 1
        whisk_width = 0.1*bar_width
        MA_width = 5*whisk_width

        green = 'green'
        red = 'red'
        facecolor = 'ghostwhite'
        plt.style.use('ggplot')

        fig= plt.figure()
        spec = gridspec.GridSpec(ncols = 1, nrows=2, hspace=0.01, height_ratios=[4,1])

        ax0 = fig.add_subplot(spec[0])
        ax0.set_facecolor(facecolor)
        ax0.bar(bullishSticks.index, bullishSticks.close - bullishSticks.open, bar_width, bottom=bullishSticks.open, color=red)
        ax0.bar(bullishSticks.index, bullishSticks.high - bullishSticks.close, whisk_width, bottom=bullishSticks.close, color=red)
        ax0.bar(bullishSticks.index, bullishSticks.low - bullishSticks.open, whisk_width, bottom=bullishSticks.open, color=red)

        ax0.bar(bearishSticks.index, bearishSticks.close - bearishSticks.open, bar_width, bottom=bearishSticks.open, color=green)
        ax0.bar(bearishSticks.index, bearishSticks.high - bearishSticks.open, whisk_width, bottom=bearishSticks.open, color=green)
        ax0.bar(bearishSticks.index, bearishSticks.low - bearishSticks.close, whisk_width, bottom=bearishSticks.close, color=green)

        for lag in lags:
            ax0.plot(MA_df[f'MA_{lag}'],linewidth=MA_width, label=f'MA_{lag}')
        ax0.legend()

        ax0.tick_params(axis='x', colors='white')
        ax0.set_ylabel('Prices')

        ax1 = fig.add_subplot(spec[1])
        ax1.set_facecolor(facecolor)
        ax1.bar(bullishSticks.index, bullishSticks.volume,color=red)
        ax1.bar(bearishSticks.index, bearishSticks.volume,color=green)
        ax1.set_ylabel('Volume')

        plt.xticks(rotation=30, ha='right')
        plt.suptitle(f'{self.ticker}(frequency: {self.frequency})') 

        save_file_path = f"{file_dir}//{self.ticker}_{self.start_date}_{self.end_date}_{self.frequency}_candle.png"
        fig.savefig(save_file_path)

        #plt.show()
        plt.close()
        return save_file_path

if __name__ == '__main__':
    
    dummy_stock = IndividualStock(ticker = '000001.XSHE',
                                start_date='2020-01-01',
                                end_date='2020-01-15',
                                frequency='1d')
    # 我尝试的拼接方法只能拼一个完整矩阵的大图（小图数量是3的倍数），因此在引入超过2个个股时考虑引入dummy使总体小图数量刚好为6或9，最后再删去dummy以达到效果

    xlsx_name = 'stocks_queries.xlsx'
    xlsx_df = pd.read_excel(f'{file_dir}/{xlsx_name}')
    #xlsx_df = xlsx_df[:1] # 调试用
    num_queries = len(xlsx_df)
    # 形成图表路径的list
    fig_dirs = []
    for i in range(len(xlsx_df)):
        stock = IndividualStock(ticker = xlsx_df.code[i],
                            start_date= xlsx_df.start_date[i],
                            end_date= xlsx_df.end_date[i],
                            frequency= xlsx_df.frequency[i])
        fig = stock.plot_candlesticks()
        fig_dirs.append(fig)

    if len(fig_dirs)> 1:
        nrow = math.ceil(len(fig_dirs)/3)
        if nrow > 1:
            # 添加冗余的dummy_stock使list的长度为3的倍数
            if nrow*3-num_queries != 0:
                for k in range(nrow*3-num_queries):
                    fig_dirs.append(dummy_stock.plot_candlesticks())
            fig, axes = plt.subplots(nrow, 3, figsize=(20, 20))
            for i, ax in enumerate(axes.flat):
                img = mpimg.imread(fig_dirs[i])
                ax.imshow(img)
                ax.axis('off')
            # 移除dummy_stock的图像
            if nrow*3-num_queries == 2:
                axes[nrow-1,-1].set_visible(False) 
                axes[nrow-1,1].set_visible(False)
            elif nrow*3-num_queries == 1:
                axes[nrow-1,-1].set_visible(False)
        else:
            fig, axes = plt.subplots(1, num_queries, figsize=(20, 20))
            for i, ax in enumerate(axes.flat):
                img = mpimg.imread(fig_dirs[i])
                ax.imshow(img)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{file_dir}/concat_chart.png')
    plt.close(fig)