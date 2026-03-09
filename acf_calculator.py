import time
import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm


def calculate_acf_5d_feature(fields_data, market_value_data, factor_name, window=5, st_data=None, industry_data=None):
    """
    计算ACF综合因子（变盘度）
    
    基于公式：ACF = -ln(1+ma_std)，其中ma_std是不同窗口期的成交额标准差（5日、10日、20日、60日、120日）
    
    Args:
        fields_data: OHLC数据 (polars.DataFrame 或 pandas.DataFrame)
        market_value_data: 市值数据 (pandas.DataFrame)
        factor_name: 因子名称 (str)
        window: 计算窗口期（天数），默认5天
        st_data: ST股票数据 (polars.DataFrame 或 pandas.DataFrame, 可选)
                 如果提供，将在计算前过滤掉ST股票
        industry_data: 行业市值数据 (pandas.DataFrame, 可选)
                       如果提供，将在市值中性化后进行行业中性化处理
                       需要包含列：datetime, vt_symbol, industry, industry_market_value
    
    Returns:
        pl.DataFrame: 计算结果 (polars DataFrame)
    """
    start_time = time.time()
    
    # 转换 fields_data 为 pandas DataFrame
    fields_df = fields_data.to_pandas() if hasattr(fields_data, 'to_pandas') else fields_data
    
    # 合并 fields_data（OHLC）和市值数据
    df = pd.merge(fields_df, market_value_data, on=["datetime", "vt_symbol"], how="left")
    df = df.sort_values(["vt_symbol", "datetime"]).reset_index(drop=True)
    
    # 过滤ST股票（如果提供了ST数据）
    if st_data is not None:
        print("步骤0: 过滤ST股票...")
        st_df = st_data.to_pandas() if hasattr(st_data, 'to_pandas') else st_data
        # 合并ST数据，标记ST股票
        df = pd.merge(df, st_df[["datetime", "vt_symbol", "st_flag"]], 
                     on=["datetime", "vt_symbol"], how="left")
        # 过滤掉ST股票（st_flag == 1），保留非ST股票（st_flag != 1 或 st_flag为NaN）
        before_count = len(df)
        df = df[(df['st_flag'] != 1) | (df['st_flag'].isna())].copy()
        df = df.drop(columns=['st_flag'], errors='ignore')  # 删除st_flag列
        after_count = len(df)
        removed_count = before_count - after_count
        print(f"  过滤前: {before_count} 条记录, 过滤后: {after_count} 条记录, 移除: {removed_count} 条ST股票记录")
    
    # 对于每个股票，计算过去window日的指标
    
    # 使用rolling窗口高效计算
    # 计算过去window日VWAP的算术平均（分子）
    df['amount']=df['close_adj_af']*df['volume']
    windows = [5, 10, 20, 60, 120]
    for w in windows:
        df[f"move_amount_{w}"] = (
        df
        .groupby("vt_symbol")["amount"]
        .transform(lambda x: x.rolling(w, min_periods=w).mean())
    )
    # 计算过去window日成交量加权的VWAP（分母）
    # sum(volume * vwap) / sum(volume)
    df['ma_std'] = df[['amount', 'move_amount_5', 'move_amount_10', 'move_amount_20', 'move_amount_60','move_amount_120']].std(axis=1)
    df['ACF']=-np.log1p(df['ma_std'])
    
    print("步骤3: 市值中性化处理...")
    # 进行市值中性化（使用线性回归残差）
    def neutralize_group(group):
        """对单个日期的数据进行中性化"""
        result = group.copy()
        
        feature_vals = group['ACF']
        neutralizer_vals = group['neg_market_value']
        
        # 过滤掉NaN值
        valid_mask = feature_vals.notna() & neutralizer_vals.notna()
        if valid_mask.sum() < 2:
            result['ACF'] = feature_vals
            return result
        
        feature_valid = feature_vals[valid_mask]
        neutralizer_valid = neutralizer_vals[valid_mask]
        
        # 进行线性回归
        try:
            X = sm.add_constant(neutralizer_valid)
            y = feature_valid
            model = sm.OLS(y, X).fit()
            residuals = model.resid
            
            # 构建结果
            result_col = feature_vals.copy()
            result_col[valid_mask] = residuals.values
            result['ACF'] = result_col
        except:
            result['ACF'] = feature_vals
        
        return result
    
    df = df.groupby('datetime', group_keys=False).apply(neutralize_group).reset_index(drop=True)
    
    # 行业中性化处理（如果提供了行业数据）
    if industry_data is not None:
        print("步骤3.5: 行业中性化处理...")
        industry_df = industry_data if isinstance(industry_data, pd.DataFrame) else industry_data.to_pandas()
        
        # 合并行业数据
        df = pd.merge(df, industry_df[["datetime", "vt_symbol", "industry"]], 
                     on=["datetime", "vt_symbol"], how="left")
        
        # 对每个日期，计算每个行业的因子均值，然后从因子值中减去行业均值
        def neutralize_industry_group(group):
            """对单个日期的数据进行行业中性化"""
            result = group.copy()
            
            # 过滤掉行业为空的股票（如果没有行业分类，则不进行行业中性化）
            valid_industry_mask = group['industry'].notna()
            
            if valid_industry_mask.sum() > 0:
                # 按行业分组，计算每个行业的因子均值
                industry_mean = group.loc[valid_industry_mask].groupby('industry')['ACF'].transform('mean')
                
                # 减去行业均值（行业中性化）
                result.loc[valid_industry_mask, 'ACF'] = group.loc[valid_industry_mask, 'ACF'] - industry_mean
                # 对于没有行业分类的股票，保持原值不变
            
            return result
        
        df = df.groupby('datetime', group_keys=False).apply(neutralize_industry_group).reset_index(drop=True)
        
        # 删除industry列（不再需要）
        df = df.drop(columns=['industry'], errors='ignore')
        print("  行业中性化完成")
    
    print("步骤4: 极值处理（99%/1%分位数截断）...")
    # 进行去极值处理
    def winsorize_group(group):
        """对单个日期的数据进行去极值"""
        result = group.copy()
        lower_bound = group['ACF'].quantile(0.01)
        upper_bound = group['ACF'].quantile(0.99)
        result['ACF'] = result['ACF'].clip(lower=lower_bound, upper=upper_bound)
        return result
    
    df = df.groupby('datetime', group_keys=False).apply(winsorize_group).reset_index(drop=True)
    
    print("步骤5: 最终横截面标准化...")
    # 最终标准化（只对APB列）
    def zscore_apb_group(group):
        """对单个日期的ACF数据进行标准化"""
        result = group.copy()
        mean_val = group['ACF'].mean()
        std_val = group['ACF'].std()
        if std_val != 0:
            result['ACF'] = (group['ACF'] - mean_val) / std_val
        else:
            result['ACF'] = 0
        return result
    
    df = df.groupby('datetime', group_keys=False).apply(zscore_apb_group).reset_index(drop=True)
    
    # 重命名为因子名称
    df = df.rename(columns={'ACF': factor_name})
    
    # 选择结果列
    result_df_pd = df[["datetime", "vt_symbol", factor_name]].copy()
    
    # 转换为 polars DataFrame
    result_df = pl.from_pandas(result_df_pd)
    
    end_time = time.time()
    print(f"ACF_{window}d因子计算完成，耗时: {end_time - start_time:.2f} 秒")
    
    return result_df
