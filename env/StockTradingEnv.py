import random
import json
import gym
import sys
from gym import spaces
import pandas as pd
import numpy as np
from scipy.special import softmax


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, settings, test):
        super(StockTradingEnv, self).__init__()

        self.settings = settings
        self.test = test

        self.df = df

        self.past_horzion = int(self.settings['past_horzion'])

        self.df.fillna(method ='pad') 
        self.df.fillna(0.) 
        self.df.replace(to_replace = np.nan, value = 0.)

        # self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min() + 1e-4)

        self.first_day = self.df.first_valid_index()


        self.action_space = spaces.Box(
            low = np.zeros(3), 
            high = np.ones(3), 
            dtype = np.float16)

        self.observation_space = spaces.Box(
            np.zeros(self.past_horzion), 
            np.ones(self.past_horzion), 
            dtype=np.float16)


    def _next_observation(self):

        observation_vec = self.df['Close'].iloc[self.current_step-self.past_horzion:self.current_step].fillna(0.).astype(np.float32)
        # observation_vec -= np.amin(observation_vec)
        # observation_vec /= np.amax(observation_vec)
        # observation_vec -= np.mean(observation_vec)
        # observation_vec /= np.var(observation_vec)

        if self.test:
            return observation_vec.astype(np.float32)
        else:
            return observation_vec.astype(np.float32) #+ np.random.normal(0., 1e-3, (len(observation_vec))).astype(np.float32)


    def _take_action(self, action):
     
        self.current_price_original = self.df['Close'].iloc[self.current_step]
        
        self.action_probs = softmax(action)
        buy_confidence = self.action_probs[0]
        sell_confidence = self.action_probs[2]

        self.action = np.argmax(self.action_probs)

        random_exploration = np.random.randint(0,3)
        if np.random.uniform(0.,1.) > .8 or self.action == 1:
            # self.action = random_exploration
            self.action = 0

        # 0-buy | 1-hold | 2-sell
        if self.action == 0:

            num_possible = np.floor((self.cash - self.settings['stop_below_balance']) / (self.current_price_original + 1e-5))

            position_size = int(np.floor(buy_confidence * num_possible))

            self.cash -= position_size * self.current_price_original
            self.cash -= self.settings['transation_fee']
            self.shares_held += position_size

            self.trades_done += 1

            if self.test: print('bought ' + str(position_size) + ' at $' + str(self.current_price_original))

        if self.action == 1:

            if self.test: print('held position')

        if self.action == 2:

            position_size = np.int(np.floor(sell_confidence * self.shares_held))

            self.cash += position_size * self.current_price_original
            self.cash -= self.settings['transation_fee']
            self.shares_held -= position_size

            self.trades_done += 1

            if self.test: print('sold ' + str(position_size) + ' at $' + str(self.current_price_original))

        self.equity = self.cash + self.shares_held * self.current_price_original
        self.value_in_shares = self.shares_held * self.current_price_original



    def step(self, action):

        self._take_action(action)

        self.current_step += 1       
        self.steps_taken += 1

        reward = self.equity - self.cash + self.shares_held #* self.current_step / self.settings['max_steps']

        done = self.equity <= self.settings['stop_below_balance'] or self.current_step > len(self.df) - 1 or self.steps_taken > self.settings['max_steps']

        if done: self.reset()

        obs = self._next_observation()

        monitor_data = {
                'equity': self.equity,
                'trades_done': self.trades_done,
                'shares_held': self.shares_held,
                'value_in_shares': self.value_in_shares,
                'cash': self.cash,
                'action': softmax(action),
                'action_prob': np.argmax(softmax(action)),
                }

        return obs, reward, done, {}, monitor_data


    def reset(self):

        self.cash = self.settings['inital_account_balance']
        self.net_worth = self.settings['inital_account_balance']

        self.shares_held = 0
        self.value_in_shares = 0

        self.trades_done = 0

        self.steps_taken = 0

        # self.equity = []
        # self.equity.append(self.settings['inital_account_balance'])
        self.equity = self.settings['inital_account_balance']

        self.transations = 0
        
        if self.test: self.current_step = self.past_horzion
        else: self.current_step = random.randint(self.past_horzion, len(self.df) - self.past_horzion - self.settings['max_steps'])

        return self._next_observation()


    def render(self, mode='human', close=False):

        profit = self.net_worth - self.settings['inital_account_balance']

        if self.current_step == 400:
      
            print('Step: ', int(self.current_step), 
                # '\nAction--->  type/amount/symbol ', self.action_type, '/ ', self.amount, '/ ', symbol,
                '\nBalance: ', np.round(self.balance, 1), 
                '\nShares held/purch./sold: ', int(np.sum(self.shares_held)), '/ ', int(np.sum(self.total_shares_purchased)), '/ ', int(np.sum(self.total_shares_sold)), 
                '\nFees: ', np.round(self.total_fees, 1),
                '\nProfit: ', np.round(profit, 1),
                end = '\n\r\n')








