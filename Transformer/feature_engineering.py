import os
import numpy as np
import pandas as pd
import polars as pl

from custom_config import cfg, set_random_seed, TXT_COLS

# https://www.kaggle.com/code/machengyuan/simple-xgb-model
CATS = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
NUMS = ['page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
        'hover_duration', 'elapsed_time_diff']

NAME_FEATURE = ['basic', 'undefined', 'close', 'open', 'prev', 'next']
EVENT_NAME_FEATURE = ['cutscene_click', 'person_click', 'navigate_click',
                      'observation_click', 'notification_click', 'object_click',
                      'object_hover', 'map_hover', 'map_click', 'checkpoint',
                      'notebook_click']

FQID_LIST = sorted(['worker', 'archivist', 'gramps', 'wells', 'toentry', 'confrontation', 'crane_ranger', 'groupconvo', 'flag_girl', 'tomap', 'tostacks', 'tobasement', 'archivist_glasses', 'boss', 'journals', 'seescratches', 'groupconvo_flag', 'cs', 'teddy', 'expert', 'businesscards', 'ch3start', 'tunic.historicalsociety', 'tofrontdesk', 'savedteddy', 'plaque', 'glasses', 'tunic.drycleaner', 'reader_flag', 'tunic.library', 'tracks', 'tunic.capitol_2', 'trigger_scarf', 'reader', 'directory', 'tunic.capitol_1', 'journals.pic_0.next', 'unlockdoor', 'tunic', 'what_happened', 'tunic.kohlcenter', 'tunic.humanecology', 'colorbook', 'logbook', 'businesscards.card_0.next', 'journals.hub.topics', 'logbook.page.bingo', 'journals.pic_1.next', 'journals_flag', 'reader.paper0.next', 'tracks.hub.deer', 'reader_flag.paper0.next', 'trigger_coffee', 'wellsbadge', 'journals.pic_2.next', 'tomicrofiche', 'journals_flag.pic_0.bingo', 'plaque.face.date', 'notebook', 'tocloset_dirty', 'businesscards.card_bingo.bingo', 'businesscards.card_1.next', 'tunic.wildlife', 'tunic.hub.slip', 'tocage', 'journals.pic_2.bingo', 'tocollectionflag', 'tocollection', 'chap4_finale_c', 'chap2_finale_c', 'lockeddoor', 'journals_flag.hub.topics', 'tunic.capitol_0', 'reader_flag.paper2.bingo', 'photo', 'tunic.flaghouse', 'reader.paper1.next', 'directory.closeup.archivist', 'intro', 'businesscards.card_bingo.next', 'reader.paper2.bingo', 'retirement_letter', 'remove_cup', 'journals_flag.pic_0.next', 'magnify', 'coffee', 'key', 'togrampa', 'reader_flag.paper1.next', 'janitor', 'tohallway', 'chap1_finale', 'report', 'outtolunch', 'journals_flag.hub.topics_old', 'journals_flag.pic_1.next', 'reader.paper2.next', 'chap1_finale_c', 'reader_flag.paper2.next', 'door_block_talk', 'journals_flag.pic_1.bingo', 'journals_flag.pic_2.next', 'journals_flag.pic_2.bingo', 'block_magnify', 'reader.paper0.prev', 'block', 'reader_flag.paper0.prev', 'block_0', 'door_block_clean', 'reader.paper2.prev', 'reader.paper1.prev', 'doorblock', 'tocloset', 'reader_flag.paper2.prev', 'reader_flag.paper1.prev', 'block_tomap2', 'journals_flag.pic_0_old.next', 'journals_flag.pic_1_old.next', 'block_tocollection', 'block_nelson', 'journals_flag.pic_2_old.next', 'block_tomap1', 'block_badge', 'need_glasses', 'block_badge_2', 'fox', 'block_1'])
TEXT_LIST = sorted(['tunic.historicalsociety.cage.confrontation', 'tunic.wildlife.center.crane_ranger.crane', 'tunic.historicalsociety.frontdesk.archivist.newspaper', 'tunic.historicalsociety.entry.groupconvo', 'tunic.wildlife.center.wells.nodeer', 'tunic.historicalsociety.frontdesk.archivist.have_glass', 'tunic.drycleaner.frontdesk.worker.hub', 'tunic.historicalsociety.closet_dirty.gramps.news', 'tunic.humanecology.frontdesk.worker.intro', 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation', 'tunic.historicalsociety.basement.seescratches', 'tunic.historicalsociety.collection.cs', 'tunic.flaghouse.entry.flag_girl.hello', 'tunic.historicalsociety.collection.gramps.found', 'tunic.historicalsociety.basement.ch3start', 'tunic.historicalsociety.entry.groupconvo_flag', 'tunic.library.frontdesk.worker.hello', 'tunic.library.frontdesk.worker.wells', 'tunic.historicalsociety.collection_flag.gramps.flag', 'tunic.historicalsociety.basement.savedteddy', 'tunic.library.frontdesk.worker.nelson', 'tunic.wildlife.center.expert.removed_cup', 'tunic.library.frontdesk.worker.flag', 'tunic.historicalsociety.frontdesk.archivist.hello', 'tunic.historicalsociety.closet.gramps.intro_0_cs_0', 'tunic.historicalsociety.entry.boss.flag', 'tunic.flaghouse.entry.flag_girl.symbol', 'tunic.historicalsociety.closet_dirty.trigger_scarf', 'tunic.drycleaner.frontdesk.worker.done', 'tunic.historicalsociety.closet_dirty.what_happened', 'tunic.wildlife.center.wells.animals', 'tunic.historicalsociety.closet.teddy.intro_0_cs_0', 'tunic.historicalsociety.cage.glasses.afterteddy', 'tunic.historicalsociety.cage.teddy.trapped', 'tunic.historicalsociety.cage.unlockdoor', 'tunic.historicalsociety.stacks.journals.pic_2.bingo', 'tunic.historicalsociety.entry.wells.flag', 'tunic.humanecology.frontdesk.worker.badger', 'tunic.historicalsociety.stacks.journals_flag.pic_0.bingo', 'tunic.historicalsociety.closet.intro', 'tunic.historicalsociety.closet.retirement_letter.hub', 'tunic.historicalsociety.entry.directory.closeup.archivist', 'tunic.historicalsociety.collection.tunic.slip', 'tunic.kohlcenter.halloffame.plaque.face.date', 'tunic.historicalsociety.closet_dirty.trigger_coffee', 'tunic.drycleaner.frontdesk.logbook.page.bingo', 'tunic.library.microfiche.reader.paper2.bingo', 'tunic.kohlcenter.halloffame.togrampa', 'tunic.capitol_2.hall.boss.haveyougotit', 'tunic.wildlife.center.wells.nodeer_recap', 'tunic.historicalsociety.cage.glasses.beforeteddy', 'tunic.historicalsociety.closet_dirty.gramps.helpclean', 'tunic.wildlife.center.expert.recap', 'tunic.historicalsociety.frontdesk.archivist.have_glass_recap', 'tunic.historicalsociety.stacks.journals_flag.pic_1.bingo', 'tunic.historicalsociety.cage.lockeddoor', 'tunic.historicalsociety.stacks.journals_flag.pic_2.bingo', 'tunic.historicalsociety.collection.gramps.lost', 'tunic.historicalsociety.closet.notebook', 'tunic.historicalsociety.frontdesk.magnify', 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo', 'tunic.wildlife.center.remove_cup', 'tunic.library.frontdesk.wellsbadge.hub', 'tunic.wildlife.center.tracks.hub.deer', 'tunic.historicalsociety.frontdesk.key', 'tunic.library.microfiche.reader_flag.paper2.bingo', 'tunic.flaghouse.entry.colorbook', 'tunic.wildlife.center.coffee', 'tunic.capitol_1.hall.boss.haveyougotit', 'tunic.historicalsociety.basement.janitor', 'tunic.historicalsociety.collection_flag.gramps.recap', 'tunic.wildlife.center.wells.animals2', 'tunic.flaghouse.entry.flag_girl.symbol_recap', 'tunic.historicalsociety.closet_dirty.photo', 'tunic.historicalsociety.stacks.outtolunch', 'tunic.library.frontdesk.worker.wells_recap', 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap', 'tunic.capitol_0.hall.boss.talktogramps', 'tunic.historicalsociety.closet.photo', 'tunic.historicalsociety.collection.tunic', 'tunic.historicalsociety.closet.teddy.intro_0_cs_5', 'tunic.historicalsociety.closet_dirty.gramps.archivist', 'tunic.historicalsociety.closet_dirty.door_block_talk', 'tunic.historicalsociety.entry.boss.flag_recap', 'tunic.historicalsociety.frontdesk.archivist.need_glass_0', 'tunic.historicalsociety.entry.wells.talktogramps', 'tunic.historicalsociety.frontdesk.block_magnify', 'tunic.historicalsociety.frontdesk.archivist.foundtheodora', 'tunic.historicalsociety.closet_dirty.gramps.nothing', 'tunic.historicalsociety.closet_dirty.door_block_clean', 'tunic.capitol_1.hall.boss.writeitup', 'tunic.library.frontdesk.worker.nelson_recap', 'tunic.library.frontdesk.worker.hello_short', 'tunic.historicalsociety.stacks.block', 'tunic.historicalsociety.frontdesk.archivist.need_glass_1', 'tunic.historicalsociety.entry.boss.talktogramps', 'tunic.historicalsociety.frontdesk.archivist.newspaper_recap', 'tunic.historicalsociety.entry.wells.flag_recap', 'tunic.drycleaner.frontdesk.worker.done2', 'tunic.library.frontdesk.worker.flag_recap', 'tunic.humanecology.frontdesk.block_0', 'tunic.library.frontdesk.worker.preflag', 'tunic.historicalsociety.basement.gramps.seeyalater', 'tunic.flaghouse.entry.flag_girl.hello_recap', 'tunic.historicalsociety.closet.doorblock', 'tunic.drycleaner.frontdesk.worker.takealook', 'tunic.historicalsociety.basement.gramps.whatdo', 'tunic.library.frontdesk.worker.droppedbadge', 'tunic.historicalsociety.entry.block_tomap2', 'tunic.library.frontdesk.block_nelson', 'tunic.library.microfiche.block_0', 'tunic.historicalsociety.entry.block_tocollection', 'tunic.historicalsociety.entry.block_tomap1', 'tunic.historicalsociety.collection.gramps.look_0', 'tunic.library.frontdesk.block_badge', 'tunic.historicalsociety.cage.need_glasses', 'tunic.library.frontdesk.block_badge_2', 'tunic.kohlcenter.halloffame.block_0', 'tunic.capitol_0.hall.chap1_finale_c', 'tunic.capitol_1.hall.chap2_finale_c', 'tunic.capitol_2.hall.chap4_finale_c', 'tunic.wildlife.center.fox.concern', 'tunic.drycleaner.frontdesk.block_0', 'tunic.historicalsociety.entry.gramps.hub', 'tunic.humanecology.frontdesk.block_1', 'tunic.drycleaner.frontdesk.block_1'])
ROOM_FQID_LIST = sorted(['tunic.historicalsociety.entry', 'tunic.wildlife.center', 'tunic.historicalsociety.cage', 'tunic.library.frontdesk', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.historicalsociety.closet_dirty', 'tunic.humanecology.frontdesk', 'tunic.historicalsociety.basement', 'tunic.kohlcenter.halloffame', 'tunic.library.microfiche', 'tunic.drycleaner.frontdesk', 'tunic.historicalsociety.collection', 'tunic.historicalsociety.closet', 'tunic.flaghouse.entry', 'tunic.historicalsociety.collection_flag', 'tunic.capitol_1.hall', 'tunic.capitol_0.hall', 'tunic.capitol_2.hall'])

# Get the text-column maps and encode the categorical columns
def get_txt_cols_encoding_maps(df, TXT_COLS):
    maps = {}
    for col in TXT_COLS:
        maps[col] = dict(zip(np.sort(df[col].unique()).tolist(), range(df[col].nunique())))
    return maps

def process_data(df, only_drop = False):
    '''
    Process the train data.
    Steps:
      1. Drop three columns of full NaN values "fullscreen, hq, music";
      2. Fill NaN values in float columns by 0.
    '''
    df
    if cfg.use_polar:
        # Drop some columns
        df = df.drop(['fullscreen', 'hq', 'music'])
        df = df.with_column(
            pl.col('page').cast(pl.Float32, strict = False)
        )
        
        if not only_drop:
            fill_null = []
            for c in df.columns:
                if df[c].dtype == pl.datatypes.Float64:
                    fill_null.append(pl.col(c).fill_null(-999.))
                elif df[c].dtype == pl.datatypes.Utf8:
                    fill_null.append(pl.col(c).fill_null(f'no {c}'))
                else:   # Maybe integers
                    fill_null.append(pl.col(c).fill_null(-999))

            df = df.with_columns(fill_null)

    else:
        # Drop some columns
        df = df.drop(['fullscreen', 'hq', 'music'], axis = 1)
        df['page'] = df['page'].astype('float32')
        
        if not only_drop:
            # Fill NaN
            for col in df.columns:
                if df[col].dtype == float:
                    df[col] = df[col].fillna(-999.)
                elif df[col].dtype == object:
                    df[col] = df[col].fillna(f'no {col}')
                else:   # Maybe integers
                    df[col] = df[col].fillna(-999)
    
            
    return df

def add_columns_pl(df):
    columns = [
        (
            (pl.col('elapsed_time') - pl.col('elapsed_time').shift(1)) 
             .fill_null(0)
             .clip(0, 1e9)
             .over(['session_id', 'level_group'])
             .alias('elapsed_time_diff')
        ),
        (
            (pl.col('screen_coor_x') - pl.col('screen_coor_x').shift(1)) 
             .abs()
             .over(['session_id', 'level_group'])
            .alias('location_x_diff') 
        ),
        (
            (pl.col('screen_coor_y') - pl.col('screen_coor_y').shift(1)) 
             .abs()
             .over(['session_id', 'level_group'])
            .alias('location_y_diff') 
        ),
        pl.col('fqid').fill_null('fqid_None'),
        pl.col('text_fqid').fill_null('text_fqid_None')
    ]
    
    return columns

def feature_engineering_pl(df, group = '(0-4)', use_extra_features = True, feature_suffix = '', remaining_features = None):
    '''
    Feature engineering with polars
    Args:
        df: (pl.DataFrame) the training data of given group.
        df_all: (pl.DataFrame) the training data containing all groups. Used to define global features.
        group: (string) must be either '(0-4)', '(5-12)', or '(13-22)'. Default: '(0-4)'
        use_extra_features: (bool) whether use extra features or not, only applied if group is '(5-12)' or '(13-22)'. Default: True
        feature_suffix: (string) suffix to the features. Default: ''.
    '''
    
    # Aggregated features
    aggs = [
        # Number of actions
        pl.col('index').count().alias(f'session_number_{feature_suffix}'),
        
        # Number of non-null values in CATS features
        *[pl.col(c).drop_nulls().n_unique().alias(f'{c}_unique_{feature_suffix}') for c in CATS],
        
        # Quantile-based features
        *[pl.col(c).quantile(0.1, 'nearest').alias(f'{c}_quantile1_{feature_suffix}') for c in NUMS],
        *[pl.col(c).quantile(0.2, 'nearest').alias(f'{c}_quantile2_{feature_suffix}') for c in NUMS],
        *[pl.col(c).quantile(0.4, 'nearest').alias(f'{c}_quantile4_{feature_suffix}') for c in NUMS],
        *[pl.col(c).quantile(0.6, 'nearest').alias(f'{c}_quantile6_{feature_suffix}') for c in NUMS],
        *[pl.col(c).quantile(0.8, 'nearest').alias(f'{c}_quantile8_{feature_suffix}') for c in NUMS],
        *[pl.col(c).quantile(0.9, 'nearest').alias(f'{c}_quantile9_{feature_suffix}') for c in NUMS],
        
        # Statistical features
        *[pl.col(c).mean().alias(f'{c}_mean_{feature_suffix}') for c in NUMS],
        *[pl.col(c).median().alias(f'{c}_median_{feature_suffix}') for c in NUMS],
        *[pl.col(c).std().alias(f'{c}_std_{feature_suffix}') for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        
        # EVENT_NAME_FEATURE counts
        *[pl.col('event_name').filter(pl.col('event_name') == c).count().alias(f'{c}_event_name_counts{feature_suffix}')for c in EVENT_NAME_FEATURE],
        
        # Elapsed time difference - quantile features - EVENT_NAME_FEATURE
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.1, 'nearest').alias(f'{c}_ET_quantile1_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.2, 'nearest').alias(f'{c}_ET_quantile2_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.4, 'nearest').alias(f'{c}_ET_quantile4_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.6, 'nearest').alias(f'{c}_ET_quantile6_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.8, 'nearest').alias(f'{c}_ET_quantile8_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).quantile(0.9, 'nearest').alias(f'{c}_ET_quantile9_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        
        # Elapsed time difference - statistical features - EVENT_NAME_FEATURE
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).mean().alias(f'{c}_ET_mean_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).median().alias(f'{c}_ET_median_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).std().alias(f'{c}_ET_std_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).max().alias(f'{c}_ET_max_{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('event_name') == c).min().alias(f'{c}_ET_min_{feature_suffix}') for c in EVENT_NAME_FEATURE],        
        
        # Location features - EVENT_NAME_FEATURE
        *[pl.col('location_x_diff').filter(pl.col('event_name') == c).mean().alias(f'{c}_ET_mean_x{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('location_x_diff').filter(pl.col('event_name') == c).median().alias(f'{c}_ET_median_x{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('location_x_diff').filter(pl.col('event_name') == c).std().alias(f'{c}_ET_std_x{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('location_x_diff').filter(pl.col('event_name') == c).max().alias(f'{c}_ET_max_x{feature_suffix}') for c in EVENT_NAME_FEATURE],
        *[pl.col('location_x_diff').filter(pl.col('event_name') == c).min().alias(f'{c}_ET_min_x{feature_suffix}') for c in EVENT_NAME_FEATURE],
        
        # NAME_FEATURE counts
        *[pl.col('name').filter(pl.col('name') == c).count().alias(f'{c}_name_counts{feature_suffix}')for c in NAME_FEATURE],   
        
        # Elapsed time difference - quantile features - NAME_FEATURE
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.1, 'nearest').alias(f'{c}_ET_quantile1_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.2, 'nearest').alias(f'{c}_ET_quantile2_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.4, 'nearest').alias(f'{c}_ET_quantile4_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.6, 'nearest').alias(f'{c}_ET_quantile6_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.8, 'nearest').alias(f'{c}_ET_quantile8_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).quantile(0.9, 'nearest').alias(f'{c}_ET_quantile9_{feature_suffix}') for c in NAME_FEATURE],
        
        # Elapsed time difference - statistical features - NAME_FEATURE
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).mean().alias(f'{c}_ET_mean_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).median().alias(f'{c}_ET_median_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).max().alias(f'{c}_ET_max_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).min().alias(f'{c}_ET_min_{feature_suffix}') for c in NAME_FEATURE],
        *[pl.col('elapsed_time_diff').filter(pl.col('name') == c).std().alias(f'{c}_ET_std_{feature_suffix}') for c in NAME_FEATURE],
        
        # FQID_LIST statistical features
        *[pl.col('fqid').filter(pl.col('fqid') == c).count().alias(f'{c}_fqid_counts{feature_suffix}')for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).std().alias(f'{c}_ET_std_{feature_suffix}') for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).mean().alias(f'{c}_ET_mean_{feature_suffix}') for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).median().alias(f'{c}_ET_median_{feature_suffix}') for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).max().alias(f'{c}_ET_max_{feature_suffix}') for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).min().alias(f'{c}_ET_min_{feature_suffix}') for c in FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('fqid') == c).sum().alias(f'{c}_ET_sum_{feature_suffix}') for c in FQID_LIST],
        
        # TEXT_LIST statistical features
        *[pl.col('text_fqid').filter(pl.col('text_fqid') == c).count().alias(f'{c}_text_fqid_counts{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).std().alias(f'{c}_ET_std_{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).mean().alias(f'{c}_ET_mean_{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).median().alias(f'{c}_ET_median_{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).max().alias(f'{c}_ET_max_{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).min().alias(f'{c}_ET_min_{feature_suffix}') for c in TEXT_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('text_fqid') == c).sum().alias(f'{c}_ET_sum_{feature_suffix}') for c in TEXT_LIST],
        
        # ROOM_FQID_LIST statistical features
        *[pl.col('room_fqid').filter(pl.col('room_fqid') == c).count().alias(f'{c}_room_fqid_counts{feature_suffix}')for c in ROOM_FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('room_fqid') == c).std().alias(f'{c}_ET_std_{feature_suffix}') for c in ROOM_FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('room_fqid') == c).mean().alias(f'{c}_ET_mean_{feature_suffix}') for c in ROOM_FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('room_fqid') == c).max().alias(f'{c}_ET_max_{feature_suffix}') for c in ROOM_FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('room_fqid') == c).min().alias(f'{c}_ET_min_{feature_suffix}') for c in ROOM_FQID_LIST],
        *[pl.col('elapsed_time_diff').filter(pl.col('room_fqid') == c).sum().alias(f'{c}_ET_sum_{feature_suffix}') for c in ROOM_FQID_LIST],
        ]
    
    if use_extra_features:
        if group == '5-12':
            extra_aggs = [
                pl.col('elapsed_time').filter((pl.col('text') == "Here's the log book.") | 
                                               (pl.col('fqid') == 'logbook.page.bingo')).apply(lambda s: s.max() - s.min()).alias('logbook_bingo_duration'),
                pl.col('index').filter((pl.col('text') == "Here's the log book.") | 
                                       (pl.col('fqid') == 'logbook.page.bingo')).apply(lambda s: s.max() - s.min()).alias('logbook_bingo_indexCount'),
                pl.col('elapsed_time').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'reader')) | 
                                              (pl.col('fqid') == 'reader.paper2.bingo')).apply(lambda s: s.max() - s.min()).alias('reader_bingo_duration'),
                pl.col('index').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'reader')) | 
                                       (pl.col('fqid') == 'reader.paper2.bingo')).apply(lambda s: s.max() - s.min()).alias('reader_bingo_indexCount'),
                pl.col('elapsed_time').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'journals')) | 
                                              (pl.col('fqid') == 'journals.pic_2.bingo')).apply(lambda s: s.max() - s.min()).alias('journals_bingo_duration'),
                pl.col('index').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'journals')) | 
                                       (pl.col('fqid') == 'journals.pic_2.bingo')).apply(lambda s: s.max() - s.min()).alias('journals_bingo_indexCount'),
            ]

        if group == '13-22':
            extra_aggs = [
                pl.col('elapsed_time').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'reader_flag')) | 
                                              (pl.col('fqid') == 'tunic.library.microfiche.reader_flag.paper2.bingo')).apply(lambda s: s.max() - s.min() if s.len() > 0 else 0).alias('reader_flag_duration'),
                pl.col('index').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'reader_flag')) | 
                                       (pl.col('fqid') == 'tunic.library.microfiche.reader_flag.paper2.bingo')).apply(lambda s: s.max() - s.min() if s.len() > 0 else 0).alias('reader_flag_indexCount'),
                pl.col('elapsed_time').filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'journals_flag')) | 
                                              (pl.col('fqid') == 'journals_flag.pic_0.bingo')).apply(lambda s: s.max() - s.min() if s.len() > 0 else 0).alias('journalsFlag_bingo_duration'),
                pl.col("index").filter(((pl.col('event_name') == 'navigate_click') & (pl.col('fqid') == 'journals_flag')) | 
                                       (pl.col('fqid') == 'journals_flag.pic_0.bingo')).apply(lambda s: s.max() - s.min() if s.len() > 0 else 0).alias('journalsFlag_bingo_indexCount'),
            ]
        else:
            extra_aggs = []
    else:
        extra_aggs = []
    
    aggs = aggs + extra_aggs
    feature_df = df.groupby(['session_id'], maintain_order = True).agg(aggs)
    feature_df = feature_df.to_pandas()
    
    if remaining_features is None:
        # Drop some features that have more than 90% of NaN values in their realizations and columns with only 1 unique value
        null_features = feature_df.isnull().sum().sort_values(ascending = False) / len(feature_df)
        drop_cols = list(null_features[null_features > 0.9].index) + ['level_group']
        for col in feature_df.columns:
            if feature_df[col].nunique() == 1:
                drop_cols.append(col)
                
        remaining_features = [c for c in feature_df.columns if c not in drop_cols]
    
    return feature_df, remaining_features
