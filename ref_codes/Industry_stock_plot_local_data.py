import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from jarvis.utils import concat_img_row, concat_img_31
from jarvis.utils import my_candle2, FOLDER, mkdir
from omk.interface import AbstractJob
from omk.events import Event
from omk.utils.const import EVENT, ProcessType
from omk.toolkit.job_tool import JobManager
from omk.core.orm_db import EnginePointer
from omk.toolkit.db_api.db_api_daily import get_stock_quote
from omk.core.orm_db.mkt_data import RQETFDailyQuote

from PIL import Image
from WindPy import w
from datetime import datetime, timedelta
from sqlalchemy.orm import Query
from sqlalchemy import and_

import matplotlib as mpl
import seaborn as sns

from warnings import filterwarnings

filterwarnings('ignore')


class IndustryStockPlot2(AbstractJob):
    def register_event(self, event_bus, job_uuid, debug):
        if datetime.today().isoweekday() not in [6, 7]:
            event_bus.add_listener(Event(
                event_type=EVENT.PM0430,
                func=self.main,
                alert=True,
                p_type=ProcessType.Jarvis,
                des='industry stock daily plotting',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
                # read_file_path=os.path.join(FOLDER.Syn_read, 'industry_stock.xlsx'),
                read_file_path=os.path.join(FOLDER.Syn_read, 'test.xlsx'),
                save_file_path=os.path.join(FOLDER.Syn_save, 'Industry_stock_original'),
                index_col='概念类型',
                start_date=datetime.today() - timedelta(600),
                end_date=datetime.today(),
                sheet_name='概念板块'
            ))

    def organise_data(self):
        if self._data.shape[0] == 0:
            raise ValueError('data is not read!')

        self._data[self._index_col] = self._data[self._index_col].fillna(method='ffill')
        # 提取HK数据
        self._hk_data = self._data[self._data['概念类型'].str.contains('中概科技|好赛道|顺周期|大消费')]
        self._data = self._data[self._data['概念类型'].str.contains('中概科技') == False]
        self._data.set_index(self._index_col, inplace=True)

        if self._return_code:
            code_list = [str(x) if len(str(x)) == 6 else '0' * (6 - len(str(x))) + str(x) for x in
                         self._data.iloc[:, -1].to_list()]
            code_list = [x + '.SZ' if x[0] in ['0', '3'] else x + '.SH' for x in code_list]
            self._data.iloc[:, -1] = code_list

        if self._cut_day > 0:
            start_date = pd.to_datetime(self._end_date - timedelta(self._cut_day)).date()

        code_list = [x.replace('SH', 'XSHG') if 'SH' in x and 'HK' not in x else x.replace('SZ', 'XSHE') for x in
                     code_list]
        code_list = [x.replace('.HK.XSHE', '.HK') if '.HK.XSHE' in x else x.replace('.HK.SH', '.HK') for x in code_list]
        code_list = [x.replace('.WI.XSHG', '.WI') if '.WI.XSHG' in x else x for x in code_list]

        returned_df = {}
        record_list = []
        # 先处理内地股票
        for code in [x for x in code_list if '.HK' not in x and '.WI' not in x]:
            counter = 1
            print(code)
            if code not in record_list:
                record_list.append(code)
                if '512000' in code or '512880' in code or '513180' in code or '516970' in code:
                    temp_data = pd.read_sql(Query(RQETFDailyQuote).filter(RQETFDailyQuote.Code.like(code)).filter(
                        and_(RQETFDailyQuote.Date >= self._start_date.date(),
                             RQETFDailyQuote.Date <= self._end_date.date())
                    ).statement,
                                            con=EnginePointer.picker('JarvisDB02')
                                            ).loc[:,
                                ['trading_date', 'rq_code', 'open', 'high', 'low', 'close', 'volume']]
                    temp_data = temp_data.set_index('trading_date')
                    code = code.replace('XSHG', 'SH') if 'XSHG' in code else code.replace('XSHE', 'SZ')
                    temp_data['section'] = self._data[self._data.iloc[:, -1] == code].index.values[0]
                    temp_data['industry'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 0].values[0]
                    temp_data['stock_name'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 1].values[0]
                else:
                    ma_df = pd.DataFrame()
                    temp_data = get_stock_quote(code,
                                                ['trading_date', 'rq_code', 'open', 'high', 'low', 'close', 'volume',
                                                 'ex_cum_factor'],
                                                'none', self._start_date, self._end_date, '1D', 'none'
                                                ).set_index(['trading_date']).replace(0, np.nan).fillna(method='ffill')
                    temp_data.ex_cum_factor = temp_data.ex_cum_factor.fillna(1.)
                    # 将价格序列进行前复权计算
                    temp_data[['open', 'high', 'low', 'close']] = (temp_data[['open', 'high', 'low', 'close']].values * \
                                                                   temp_data['ex_cum_factor'].to_frame().values) / \
                                                                  temp_data['ex_cum_factor'].iloc[0]

                    code = code.replace('XSHG', 'SH') if 'XSHG' in code else code.replace('XSHE', 'SZ')
                    temp_data['section'] = self._data[self._data.iloc[:, -1] == code].index.values[0]
                    temp_data['industry'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 0].values[0]
                    temp_data['stock_name'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 1].values[0]
                    temp_data.index = [x.date() for x in temp_data.index]
                ma_df['MA_5'] = temp_data.close.rolling(5).mean()
                ma_df['MA_10'] = temp_data.close.rolling(10).mean()
                ma_df['MA_20'] = temp_data.close.rolling(20).mean()
                ma_df['MA_30'] = temp_data.close.rolling(30).mean()
                ma_df['MA_60'] = temp_data.close.rolling(60).mean()
                ma_df['MA_120'] = temp_data.close.rolling(120).mean()
                ma_df['MA_250'] = temp_data.close.rolling(250).mean()
                if all(ma_df['MA_250'].isna()):
                    ma_df.drop(columns=['MA_250'], inplace=True)
                returned_df[code] = {'df': temp_data.loc[start_date:], 'ma_df': ma_df.loc[start_date:]}
            else:
                print(code)
                record_list.append(code + '_' + str(counter))
                if '512000' in code or '512880' in code or '513180' in code or '516970' in code:
                    temp_data = pd.read_sql(Query(RQETFDailyQuote).filter(RQETFDailyQuote.Code.like(code)).filter(
                        and_(RQETFDailyQuote.Date >= self._start_date.date(),
                             RQETFDailyQuote.Date <= self._end_date.date())
                    ).statement, con=EnginePointer.picker('JarvisDB02')
                                            ).loc[:,
                                ['trading_date', 'rq_code', 'open', 'high', 'low', 'close', 'volume']]
                    temp_data = temp_data.set_index('trading_date')
                    code = code.replace('XSHG', 'SH') if 'XSHG' in code else code.replace('XSHE', 'SZ')
                    temp_data['section'] = self._data[self._data.iloc[:, -1] == code].index.values[1]
                    temp_data['industry'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 0].values[1]
                    temp_data['stock_name'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 1].values[1]
                else:
                    ma_df = pd.DataFrame()
                    temp_data = get_stock_quote(code,
                                                ['trading_date', 'rq_code', 'open', 'high', 'low', 'close', 'volume',
                                                 'ex_cum_factor'],
                                                'none', self._start_date, self._end_date, '1D', 'none'
                                                ).set_index(['trading_date'])
                    temp_data.ex_cum_factor = [1. if x == 0 else x for x in temp_data.ex_cum_factor.values]
                    # 将价格序列进行前复权计算
                    temp_data[['open', 'high', 'low', 'close']] = (temp_data[['open', 'high', 'low', 'close']].values * \
                                                                   temp_data['ex_cum_factor'].to_frame().values) / \
                                                                  temp_data['ex_cum_factor'].iloc[0]

                    code = code.replace('XSHG', 'SH') if 'XSHG' in code else code.replace('XSHE', 'SZ')
                    temp_data['section'] = self._data[self._data.iloc[:, -1] == code].index.values[1]
                    temp_data['industry'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 0].values[1]
                    temp_data['stock_name'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 1].values[1]
                    temp_data.index = [x.date() for x in temp_data.index]

                ma_df['MA_5'] = temp_data.close.rolling(5).mean()
                ma_df['MA_10'] = temp_data.close.rolling(10).mean()
                ma_df['MA_20'] = temp_data.close.rolling(20).mean()
                ma_df['MA_30'] = temp_data.close.rolling(30).mean()
                ma_df['MA_60'] = temp_data.close.rolling(60).mean()
                ma_df['MA_120'] = temp_data.close.rolling(120).mean()
                ma_df['MA_250'] = temp_data.close.rolling(250).mean()
                if all(ma_df['MA_250'].isna()):
                    ma_df.drop(columns=['MA_250'], inplace=True)
                returned_df[code + '_' + str(counter)] = {'df': temp_data.loc[start_date:],
                                                          'ma_df': ma_df.loc[start_date:]}
                counter += 1

        # 在处理HK
        self._hk_data.set_index(self._index_col, inplace=True)
        for code in self._hk_data['股票代码'].unique():
            ma_df = pd.DataFrame()
            temp_data = w.wsd('%s' % code, "open,high,low,close,volume", "%s" % self._start_date.strftime('%Y-%m-%d'),
                              "%s" % self._end_date.strftime('%Y-%m-%d'), "")
            temp_data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times, columns=temp_data.Fields)
            temp_data.columns = temp_data.columns.str.lower()

            # w.wsq提取最后一个数据
            last_data = w.wsq("%s" % code, "rt_open,rt_high,rt_low,rt_last,rt_vol")
            last_data = pd.DataFrame(np.transpose(last_data.Data), index=[last_data.Times[0].date()],
                                     columns=last_data.Fields)
            last_data.columns = ['open', 'high', 'low', 'close', 'volume']

            # 合并
            if temp_data.index[-1] == last_data.index[0]:
                temp_data.drop(index=last_data.index, inplace=True)
            temp_data = pd.concat([temp_data, last_data], axis=0)

            temp_data['section'] = self._hk_data[self._hk_data.iloc[:, -1] == code].index.values[0]
            temp_data['industry'] = self._hk_data[self._hk_data.iloc[:, -1] == code].iloc[:, 0].values[0]
            temp_data['stock_name'] = self._hk_data[self._hk_data.iloc[:, -1] == code].iloc[:, 1].values[0]
            temp_data.index = [pd.to_datetime(x).date() for x in temp_data.index]

            ma_df['MA_5'] = temp_data.close.rolling(5).mean()
            ma_df['MA_10'] = temp_data.close.rolling(10).mean()
            ma_df['MA_20'] = temp_data.close.rolling(20).mean()
            ma_df['MA_30'] = temp_data.close.rolling(30).mean()
            ma_df['MA_60'] = temp_data.close.rolling(60).mean()
            ma_df['MA_120'] = temp_data.close.rolling(120).mean()
            ma_df['MA_250'] = temp_data.close.rolling(250).mean()
            if all(ma_df['MA_250'].isna()):
                ma_df.drop(columns=['MA_250'], inplace=True)
            returned_df[code] = {'df': temp_data.loc[start_date:], 'ma_df': ma_df.loc[start_date:]}

        return returned_df

    def organise_plot(self, returned_df):
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        sns.set(font_scale=1.5, font='SimHei')

        if len(returned_df) == 0:
            raise ValueError('returnded_df is Empty!')

        for code in returned_df:
            print(code)
            temp_df = returned_df[code]['df'].dropna()
            temp_ma = returned_df[code]['ma_df'].reindex(index=temp_df.index)

            try:
                mkdir(os.path.join(self._save_file_path, pd.to_datetime(temp_df.index.max()).strftime('%Y-%m-%d')))
            except Exception:
                pass

            # print(temp_ma)
            if '.SZ' in code and '.SZ_' not in code:
                code = code.replace('.SZ', '')
            elif '.SH' in code and '.SH_' not in code:
                code = code.replace('.SH', '')
            elif '.SZ_' in code:
                code = code.replace('.SZ_', '')
            elif '.SH_' in code:
                code = code.replace('.SH_', '')
            print(temp_df.industry.unique()[0],)
            fig = my_candle2(temp_df, temp_ma, title='%s-%s-[%s]:%s day:%s' % (
                temp_df.industry.unique()[0],
                temp_df.stock_name.unique()[0],
                code, str(round((temp_df.close[-1] - temp_df.close[-2]) / temp_df.close[-2] * 100, 2)) + '%',
                temp_df.index.max().strftime('%Y-%m-%d')), x_label='Date', y_label1='Bar', y_label2='Volume')

            fig.savefig(os.path.join(self._save_file_path, temp_df.index.max().strftime('%Y-%m-%d'), '%s_%s_%s.jpg' % (
                temp_df.section.unique()[0],
                temp_df.industry.unique()[0],
                temp_df.stock_name.unique()[0])
                                     )
                        )
            plt.close(fig)

    def concate_plot(self, concat_path=None):
        if concat_path is None:
            concat_path_original = os.path.join(FOLDER.Syn_save, 'Industry_stock_original')
            # 获取当前路径下最大日期
            for root, dirs, files in os.walk(concat_path_original):
                break
            max_date = max(dirs)
            if max_date is None:
                raise ValueError('%s path is empty!' % concat_path_original)
            concat_path = os.path.join(FOLDER.Syn_save, 'Industry_stock_concate', max_date)

            mkdir(concat_path)
        else:
            mkdir(concat_path)
        files_path = []
        for root, dirs, files in os.walk(
                os.path.join(self._save_file_path, '%s' % max_date)):
            for file in files:
                files_path.append(os.path.join(root, file))
        # print(files_path)

        file_section = {}
        for file in files_path:
            ind = file.split('\\')[-1].split('_')[0]
            if ind not in file_section:
                file_section[ind] = [file]
            else:
                file_section[ind].append(file)

        for ind in file_section:
            # ind='周期'
            files = file_section[ind]
            if len(files) > 1:
                concat_file = []
                start = 0
                end = 3
                counter = 1
                while 1:
                    if end >= len(files):
                        concat_img_31(files[start:end], concat_path,
                                      '%s_%d' % (ind, counter))
                        concat_file.append(
                            os.path.join(concat_path, '%s_%d.png' % (ind, counter)))
                        break
                    else:
                        concat_img_31(files[start:end], concat_path,
                                      '%s_%d' % (ind, counter))
                        concat_file.append(
                            os.path.join(concat_path, '%s_%d.png' % (ind, counter)))
                    start = end
                    end += 3
                    counter += 1

                concat_img_row(concat_file, concat_path, '%s_concat' % ind)
            else:
                try:
                    img = Image.open(files[0])
                    img.save(os.path.join(concat_path, '%s_concat.png' % ind))
                except Exception as e:
                    print(files[0], '不存在于路径下!')
                    continue

    def main(self, read_file_path, save_file_path=FOLDER.Syn_save, index_col=None,
             start_date=None, end_date=None, sheet_name=None, return_code=True, cut_day=250,
             picker_name='JarvisDB02'):
        w.start()
        if sheet_name is None:
            self._data = pd.read_excel(read_file_path)
        else:
            self._data = pd.read_excel(read_file_path, sheet_name=sheet_name)

        self._start_date = pd.to_datetime(start_date)
        self._end_date = pd.to_datetime(end_date)
        self._index_col = index_col
        self._return_code = return_code
        self._cut_day = cut_day
        self._save_file_path = save_file_path

        if picker_name is None:
            self.engine = EnginePointer.get_engine()
        else:
            self.engine = EnginePointer.picker(picker_name)
        return_dict = self.organise_data()
        self.organise_plot(return_dict)
        self.concate_plot()


if __name__ == '__main__':
    # from jarvis.jobs.jarvis_xmind_auto.Industry_stock_plot_local_data import IndustryStockPlot2
    # from omk.utils.const import JobType
    # JobManager.install_job('industry stock daily plotting', IndustryStockPlot2, JobType.Module, activate=True)
    model = IndustryStockPlot2()
    manager = JobManager(alert=True)
    model.register_event(event_bus=manager.event_bus, job_uuid=None, debug=False)
    manager.event_bus.event_queue_reload(EVENT.PM0430)
    manager.event_bus.sequential_publish()
