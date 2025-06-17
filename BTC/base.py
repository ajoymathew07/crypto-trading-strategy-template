import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Processes time series data to generate trading signals:
    -1 (sell), 0 (hold), or 1 (buy).

    Attributes:
        data: Price/market data for analysis
        row: Current row index being processed
        signals: np.array of generated trading signals
        position: Current position of the strategy
        prices: entry prices of the strategy(0 when not in market)
        priceColumn : column name containing prices on which strategy trades
    """
    data: pd.DataFrame
    row: int
    signals: List[int]
    position: List[int]
    prices : List[float]
    priceColumn : str

    def __init__(self, data: pd.DataFrame, priceColumn : str, name: str = '') -> None:
        """
        Initialize strategy with market data.

        Args:
            data: DataFrame with market data
            name: Optional strategy identifier
        """
        self.data = data
        self.row = 0
        self.positions = []
        self.signals = []
        self.prices = []
        self.priceColumn = priceColumn
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
    

    @abstractmethod
    def calculateSignal(self, **kwargs) -> int:
        """
        Generate trading signal for current row.

        Returns:
            int: -1 (sell), 0 (hold), or 1 (buy)
        """
        pass

    def generateSignal(self) -> int:
        signal = self.calculateSignal()
        self.signals.append(signal)
        if self.positions : self.positions.append(self.positions[-1]+self.signals[-1])
        else: self.positions.append(self.signals[-1])
        self.updateEntryPrice(self.data.iloc[self.row])
        self.row+=1
        return signal
    
    def updateEntryPrice(self, row : pd.Series):
        """Updates entry price"""
        if self.positions[-1]==self.signals[-1]:
            self.prices.append(row[self.priceColumn])
        else:
            if self.positions[-1] != 0: self.prices.append(self.prices[-1])
            else : self.prices.append(0)

    def executeStrategy(self) -> None:
        """Process all data rows and generate signals."""
        for _, _ in self.data.iterrows():
            self.generateSignal()

    def getSignals(self) -> np.array:
        """
        Get all trading signals, generating them if needed.

        Returns:
            np.array[int]: Complete np.array of trading signals
        """
        if not self.signals:
            self.executeStrategy()
        return np.array(self.signals)
    
    def getPositions(self) -> np.array:
        """
        Get positions held by the strategy, generating them if needed.

        Returns:
            np.array[int]: Complete np.array of positions
        """ 
        if not self.positions:
            self.executeStrategy()
        return np.array(self.positions)
    def getTradeTypes(self) -> np.array:
        """
        Get tradetypes corresponding to each trade, from signals and positions, generating them if needed.

        Returns:
            np.array[int]: Complete np.array of positions
            1  : entry into a long position, or closing an existing short position
            -1 : entry into short position, or closing an existing long position
            2  : closing a running short trade, and simultaneously intiaing a long one
            -2 : closing a long trade, and simultaneously initiating a short one
        """
        return self.getSignals()
    
    def getEntryPrices(self) -> np.array:
        """
        Get all entry prices, rerunning the strategy if needed.

        Returns:
            np.array[float]: Complete np.array of entry prices
        """
        if not self.prices:
            self.executeStrategy()
        return np.array(self.prices)

    def getPrices(self) -> np.array:
        """
        Get all close prices.

        Returns:
            np.array[float]: Complete np.array of close prices
        """
        return np.array(self.data[self.priceColumn])

class PseudoStrategy(Strategy):
    """Utility class to mimic a strategy class when only gnererated data is available."""
    def __init__(self, data, priceColumn, signals, name = ''):
        super().__init__(data, priceColumn, name)
        self._signals = signals
    
    def calculateSignal(self, **kwargs):
        return self._signals[self.row]

    @classmethod
    def loadFromFile(cls, data_file, priceColumn, signalColumn):
        data = pd.read_csv(data_file)
        return cls(data, priceColumn, data[signalColumn])
