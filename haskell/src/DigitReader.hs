module Main (main) where

import AI.HNN.FF.Network
import Data.List (maximumBy)
import Data.Ord (comparing)
import Data.Word (Word8)
import MnistLoader
import Numeric.LinearAlgebra
import Text.Printf


main :: IO ()
main = do
  (samples, validation) <- load_data_wrapper "../data/"
  net <- createNetwork 784 [30] 10
  let net' = trainNTimes 30 3.0 sigmoid sigmoid' net samples
      result = map (\(a, b) -> (output net' sigmoid a, b)) validation
  printf "%.2f%% correct\n" (100 * validate result)


validate :: [(Vector Double, Word8)] -> Float
validate results =
  correct / (fromIntegral $ length results)
  where
    correct = sum [ 1 | (x, y) <- results, maxListIndex x == y]
    maxListIndex :: Vector Double -> Word8
    maxListIndex = fst . maxPair . toList
    maxPair :: [Double] -> (Word8, Double)
    maxPair = maximumBy (comparing snd) . zip [0..]
