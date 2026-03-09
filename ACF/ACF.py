import os
import pandas as pd
from acf_calculator import calculate_acf_5d_feature
from dfa import Factor
from yzutil import YzDataClient


class ACFactor(Factor):
    """
    ACF因子：成交额不同窗口期的标准差（5日、10日、20日、60日、120日）
    衡量变盘度
    """
    
    def __init__(self, **kwargs):
        """初始化ACF因子"""
        # 确保获取完整的OHLC数据（通过 fields 参数）
        if 'fields' not in kwargs or kwargs['fields'] is None:
            kwargs['fields'] = ["open_adj_af", "high_adj_af", "low_adj_af", "close_adj_af", "volume"]
                # 显式设置缓存路径，与父类保持一致
        super().__init__(**kwargs)
    
    def get_marketvalue_data(self):
        """
        获取市值数据（database为datayes），返回 pandas.DataFrame
        支持缓存机制，避免重复获取数据
        """
        # 设置缓存目录和文件
        cache_dir = os.path.join(self.cache_path, "market_value_data")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"datayes_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.parquet")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f"从缓存加载市值数据: {cache_file}")
            market_value = pd.read_parquet(cache_file)
            return market_value[["datetime", "vt_symbol", "neg_market_value"]]
        
        print("正在获取流动市值数据...")
        # 根据start_date和end_date动态生成年份范围
        start_year = self.start_date.year
        end_year = self.end_date.year
        years = range(start_year, end_year + 1)
        market_value_list = []
        
        for year in years:
            # 动态生成每年的起止日期
            year_start_date = f"{year}-01-01"
            year_end_date = f"{year}-12-31"
            
            # 调用API获取数据并添加到列表
            market_value = self.data_api.get_equity_mkt(
                start_date=year_start_date,
                end_date=year_end_date,
                fields=['neg_market_value'],
                db='datayes'
            )
            market_value_list.append(market_value)
        
        # 合并所有年份的数据
        market_value = pd.concat(market_value_list)
        
        # 规范时间与标的列名
        market_value.rename(columns={'date': 'datetime', 'full_symbol': 'vt_symbol'}, inplace=True)
        market_value['datetime'] = pd.to_datetime(market_value['datetime'])
        market_value['vt_symbol'] = market_value['vt_symbol'].str.replace('.XSHG', '.SH').str.replace('.XSHE', '.SZ')
        
        # 保存到缓存
        market_value.to_parquet(cache_file)
        print(f"市值数据已缓存至: {cache_file}")
        
        return market_value[["datetime", "vt_symbol", "neg_market_value"]]
    
    def get_industry_market_value_data(self, method="SW2021", level=1):
        """
        获取行业市值数据，返回 pandas.DataFrame
        包含每个股票每日的行业分类和行业总市值
        
        Args:
            method: 行业分类方式，默认"SW2021"（申万2021）
            level: 行业等级，默认1（一级行业）
        
        Returns:
            pandas.DataFrame: 包含 datetime, vt_symbol, industry, industry_market_value 列
        """
        # 设置缓存目录和文件
        cache_dir = os.path.join(self.cache_path, "industry_data")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{method}_level{level}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.parquet")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f"从缓存加载行业数据: {cache_file}")
            industry_data = pd.read_parquet(cache_file)
            return industry_data
        
        print(f"正在获取行业数据（方法={method}, 等级={level}）...")
        
        # 获取交易日列表
        trade_days = self.trade_days
        
        # 获取市值数据
        market_value_df = self.get_marketvalue_data()
        
        # 存储每日的行业数据
        all_industry_data = []
        
        # 按日期获取行业数据（需要每日获取，因为行业分类可能变化）
        from tqdm import tqdm
        for trade_day in tqdm(trade_days, desc="获取行业数据"):
            trade_day_str = trade_day if isinstance(trade_day, str) else trade_day.strftime('%Y-%m-%d')
            trade_day_dt = pd.to_datetime(trade_day_str)
            
            # 获取当日行业数据
            industry_info = self.data_api.get_equity_industry(
                date=trade_day_str,
                method=method,
                level=level,
                db='wind'
            )
            
            if industry_info is not None and not industry_info.empty:
                # 规范列名
                industry_info.rename(columns={'full_symbol': 'vt_symbol'}, inplace=True)
                industry_info['vt_symbol'] = industry_info['vt_symbol'].str.replace('.XSHG', '.SH').str.replace('.XSHE', '.SZ')
                industry_info['datetime'] = trade_day_dt
                
                # 只保留需要的列
                industry_info = industry_info[['datetime', 'vt_symbol', 'industry']].copy()
                
                # 合并当日市值数据
                day_market_value = market_value_df[market_value_df['datetime'] == trade_day_dt].copy()
                
                if not day_market_value.empty:
                    # 合并行业和市值数据
                    day_data = pd.merge(industry_info, day_market_value[['vt_symbol', 'neg_market_value']], 
                                       on='vt_symbol', how='inner')
                    
                    # 计算每个行业的市值（行业总市值）
                    industry_mv = day_data.groupby(['datetime', 'industry'])['neg_market_value'].sum().reset_index()
                    industry_mv.rename(columns={'neg_market_value': 'industry_market_value'}, inplace=True)
                    
                    # 合并回原数据
                    day_data = pd.merge(day_data[['datetime', 'vt_symbol', 'industry']], 
                                      industry_mv[['datetime', 'industry', 'industry_market_value']],
                                      on=['datetime', 'industry'], how='left')
                    
                    all_industry_data.append(day_data)
                else:
                    # 如果当日没有市值数据，仍然保存行业分类信息（但不计算行业市值）
                    industry_info_only = industry_info[['datetime', 'vt_symbol', 'industry']].copy()
                    industry_info_only['industry_market_value'] = None
                    all_industry_data.append(industry_info_only)
        
        # 合并所有日期的数据
        if all_industry_data:
            industry_data = pd.concat(all_industry_data, ignore_index=True)
        else:
            print("警告：未获取到任何行业数据")
            return pd.DataFrame(columns=['datetime', 'vt_symbol', 'industry', 'industry_market_value'])
        
        # 保存到缓存
        industry_data.to_parquet(cache_file)
        print(f"行业数据已缓存至: {cache_file}")
        
        return industry_data[['datetime', 'vt_symbol', 'industry', 'industry_market_value']]
    
    def calculate_feature(self, window=5):
        """
        计算ACF综合因子
        
        使用独立计算函数进行计算
        
        Args:
            window: 计算窗口期（天数），默认5天
        """
        # 获取市值数据用于中性化
        market_value_df = self.get_marketvalue_data()
        
        # 获取行业市值数据用于行业中性化
        #industry_mv_df = self.get_industry_market_value_data()
        industry_mv_df = None
        
        # 更新因子名称以反映窗口期
        original_name = self.name
        # 如果因子名称不包含窗口期信息，则更新它
        if f"_{window}d" not in self.name:
            # 尝试从名称中提取基础名称，然后添加窗口期
            import re
            base_name = re.sub(r'_\d+d$', '', self.name)
            self.name = f"{base_name}_{window}d"
    
        # 调用独立计算函数，传递ST数据和行业数据
        result_df = calculate_acf_5d_feature(
            fields_data=self.fields_data,
            market_value_data=market_value_df,
            factor_name=self.name,
            window=window,
            st_data=self.st_data,  # 传递ST数据，用于过滤ST股票
            industry_data=industry_mv_df  # 传递行业市值数据，用于行业中性化
        )
        
        self.result_df = result_df
        
        # 恢复原始名称（如果需要）
        # self.name = original_name
        
        return self.result_df

if __name__ == '__main__':
    # 严格遵循 example_usage.py 格式
    
    # 1. 导入模块（已在文件开头完成）
    
    # 2. 创建数据客户端
    username = "intern@ohfi.com.cn"
    password = "Intern@678"
    host = "WestLake"
    data_client = YzDataClient(username, password, host)
    
    # 3. 创建因子实例（与 example_usage.py 格式完全一致）
    factor = ACFactor(
        name="acf",
        description="ACF因子（均线收敛度）",
        author="yangjunkai",
        email="yangjunkai@ohfi.com.cn",
        data_api=data_client,
        fields=["open_adj_af", "high_adj_af", "low_adj_af", "close_adj_af", "volume"],  # 显式指定
        start_date="2019-01-01",
        end_date="2025-12-31",
        benchmark_symbol = "000300.SH",
        path="."
    )
    
    print('因子实例化成功')
    
    # 4. 计算因子（使用自定义的 calculate_feature）
    # 可以通过设置window参数来选择计算几日的APB因子
    # 例如：window=5 计算5日APB，window=10 计算10日APB，等等
     # 可以修改这个值：5, 10, 15, 20, 25, 30 等
    factor.calculate_feature()
    
    # 5. 获取因子摘要（使用父类方法）
    summary = factor.get_factor_summary()
    print("因子摘要：")
    print(summary)
    
    # 6. 保存结果（使用父类方法）
    factor.save_all()
    
    # 7. 绘制图表（使用父类方法）
    factor.plot_returns()
    #factor.plot_returns_detailed()
