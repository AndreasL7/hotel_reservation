import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

class TweakFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        
        return self
    
    def pipe_logger(self, df, addon_msg=''):
        if addon_msg != '':
            logger.info(addon_msg)
        logger.info(f"Shape of DataFrame: {df.shape[0]} x {df.shape[1]}")
        return df
    
    def get_faulty_date_index(self, X: pd.DataFrame) -> pd.Index:
        
        return (X
                .loc[(X['arrival_year'] == 2018) & (X['arrival_month'] == 2) & (X['arrival_date'] == 29)]
                .index)
        
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        
        return (X
                .pipe(self.pipe_logger)
                
                .drop(columns=['Booking_ID'])
                .pipe(lambda df_: df_.drop(self.get_faulty_date_index(df_))
                .pipe(self.pipe_logger, "After dropping leap year error...")
                .assign(date_combined=lambda df_: pd.to_datetime(df_['arrival_year'].astype(str) + '-' + df_['arrival_month'].astype(str) + '-' + df_['arrival_date'].astype(str)),
                        lead_time_category=lambda df_: np.where(df_['lead_time'].lt(90), 'Within 3 months', np.where(df_['lead_time'].lt(180), 'Within 6 months', 'Beyond 6 months')),
                        is_weekend_arrival=lambda df_: np.where(df_['date_combined'].dt.weekday.gt(5), 1, 0),
                        total_people=lambda df_: df_['no_of_adults'].add(df_['no_of_children']),
                        is_alone=lambda df_: np.where(df_['total_people'].eq(1), 1, 0),
                        total_nights_stay=lambda df_: df_['no_of_weekend_nights'].add(df_['no_of_week_nights']),
                        walk_in=lambda df_: np.where(df_['lead_time'].eq(0), 1, 0),
                        promotional_offer=lambda df_: np.where(df_['avg_price_per_room'].eq(0), 1, 0),
                        arrival_weekday=lambda df_: df_['date_combined'].dt.day_name(),
                        arrival_quarter=lambda df_: df_['date_combined'].dt.quarter,
                        week_of_year=lambda df_: df_['date_combined'].dt.isocalendar().week,)
                .drop(columns=['date_combined', 'arrival_year'])
                .astype({**{k: 'int8' 
                            for k in ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                                      'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests', 'total_people', 'total_nights_stay',
                                      'arrival_month', 'arrival_date', 'arrival_quarter', 'week_of_year',
                                      'walk_in', 'promotional_offer', 'is_weekend_arrival', 'is_alone', 
                                      'required_car_parking_space', 'repeated_guest']},
                         **{k: 'int16'
                            for k in ['lead_time']},
                         **{k: 'category'
                            for k in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'lead_time_category', 
                                      'arrival_weekday']},
                         'avg_price_per_room': 'float16',})
                ))