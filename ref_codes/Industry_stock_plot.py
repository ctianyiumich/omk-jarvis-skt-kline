import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from jarvis.utils import concat_img_row, concat_img_31
from jarvis.utils import my_candle2, FOLDER, mkdir
from omk.toolkit.calendar_ import get_previous_trading_date
from omk.interface import AbstractJob
from omk.events import Event, EventBus
from omk.utils.const import EVENT, ProcessType
from omk.toolkit.job_tool import JobManager

from PIL import Image
from WindPy import w
from datetime import datetime, timedelta

import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False
sns.set(font_scale=1.5, font='SimHei')

from warnings import filterwarnings

filterwarnings('ignore')


class IndustryStockPlot(AbstractJob):

    def register_event(self, event_bus, job_uuid, debug):
        if datetime.today().isoweekday() not in [6, 7]:

            event_bus.add_listener(Event(
                event_type=EVENT.PM0415,
                func=self.main,
                alert=True,
                p_type=ProcessType.Jarvis,
                des='industry index daily plotting2',
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
                # read_file_path=os.path.join(FOLDER.Syn_read, '板块跟踪列表.xlsx'),
                # save_file_path=os.path.join(FOLDER.Syn_save, 'Industry_stock_original'),
                read_file_path=os.path.join(FOLDER.Syn_read, '板块跟踪列表.xlsx'),
                save_file_path=os.path.join(FOLDER.Syn_save, 'Industry_stock_original2'),
                index_col='概念类型',
                start_date=datetime.today() - timedelta(400),
                end_date=datetime.today(),
                sheet_name='概念板块',
                return_code=False,
            ))

    def organise_data(self):
        if self._data.shape[0] == 0:
            raise ValueError('data is not read!')

        self._data[self._index_col] = self._data[self._index_col].fillna(method='ffill')
        self._data.set_index(self._index_col, inplace=True)

        if self._return_code:
            code_list = [str(x) if len(str(x)) == 6 else '0' * (6 - len(str(x))) + str(x) for x in
                         self._data.iloc[:, -1].unique()]
            code_list = [x + '.SZ' if x[0] in ['0', '3'] else x + '.SH' for x in code_list]
            self._data.iloc[:, -1] = code_list
        else:
            code_list=self._data.iloc[:, -1].to_list()


        if self._cut_day > 0:
            start_date = pd.to_datetime(self._end_date - timedelta(self._cut_day)).date()
        returned_df = {}
        for code in code_list:
            ma_df = pd.DataFrame()
            temp_data = w.wsd("%s" % code, "high,low,close,open,volume", "%s" % self._start_date.strftime('%Y-%m-%d'),
                              "%s" % get_previous_trading_date(self._end_date.strftime('%Y-%m-%d')), "")
            temp_data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times, columns=temp_data.Fields)
            temp_data.columns = temp_data.columns.str.lower()

            temp_data2 = w.wsq("%s" % code, "rt_open,rt_high,rt_low,rt_last,rt_vol")
            temp_data2 = pd.DataFrame(np.transpose(temp_data2.Data), index=[temp_data2.Times[0].date()],
                                      columns=['open', 'high', 'low', 'close', 'volume'])[temp_data.columns]

            temp_data = pd.concat([temp_data, temp_data2], axis=0)
            temp_data['section'] = self._data[self._data.iloc[:, -1] == code].index.values[0]
            temp_data['industry'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 0].values[0]
            temp_data['stock_name'] = self._data[self._data.iloc[:, -1] == code].iloc[:, 1].values[0]

            ma_df['MA_5'] = temp_data.close.rolling(5).mean()
            ma_df['MA_10'] = temp_data.close.rolling(10).mean()
            ma_df['MA_20'] = temp_data.close.rolling(20).mean()
            ma_df['MA_30'] = temp_data.close.rolling(30).mean()
            ma_df['MA_60'] = temp_data.close.rolling(60).mean()
            ma_df['MA_120'] = temp_data.close.rolling(120).mean()
            ma_df['MA_250'] = temp_data.close.rolling(250).mean()
            returned_df[code] = {'df': temp_data.loc[start_date:], 'ma_df': ma_df.loc[start_date:]}

        return returned_df

    def organise_plot(self, returned_df):

        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        sns.set(font_scale=1.5, font='SimHei')

        if len(returned_df) == 0:
            raise ValueError('returnded_df is Empty!')

        for code in returned_df:
            temp_df = returned_df[code]['df']
            temp_ma = returned_df[code]['ma_df']

            mkdir(os.path.join(self._save_file_path, temp_df.index.max().strftime('%Y-%m-%d')))

            # print(temp_ma)
            if '.SZ' in code:
                code = code.replace('.SZ', '')
            elif '.SH' in code:
                code = code.replace('.SH', '')
            else:
                pass
            fig = my_candle2(temp_df, temp_ma, title='%s-%s-[%s]:%s day:%s' % (
                temp_df.industry.unique()[0],
                temp_df.stock_name.unique()[0],
                code, str(round((temp_df.close[-1] - temp_df.close[-2]) / temp_df.close[-2] * 100,2)) + '%',
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
            concat_path = os.path.join(FOLDER.Syn_save, 'Industry_stock_concate2',
                                       datetime.today().strftime('%Y-%m-%d'))
            mkdir(concat_path)
        else:
            mkdir(concat_path)
        files_path = []
        for root, dirs, files in os.walk(
                os.path.join(self._save_file_path, '%s' % datetime.today().strftime('%Y-%m-%d'))):
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
                img = Image.open(files[0])
                img.save(os.path.join(concat_path, '%s_concat.png' % ind))

    def main(self,read_file_path, save_file_path=FOLDER.Syn_save, index_col=None,
                 start_date=None, end_date=None, sheet_name=None, return_code=True, cut_day=90):
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

        return_dict = self.organise_data()
        self.organise_plot(return_dict)
        self.concate_plot()


if __name__ == '__main__':
    # from jarvis.jobs.jarvis_xmind_auto.Industry_stock_plot import IndustryStockPlot
    # from omk.utils.const import JobType
    # JobManager.install_job('industry stock daily plotting2', IndustryStockPlot, JobType.Module, activate=True)
    model = IndustryStockPlot()
    manager = JobManager(alert=True)
    model.register_event(event_bus=manager.event_bus, job_uuid=None, debug=False)
    manager.event_bus.event_queue_reload(EVENT.PM0415)
    manager.event_bus.sequential_publish()
