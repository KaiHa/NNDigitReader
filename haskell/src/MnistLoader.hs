{-# LANGUAGE BangPatterns #-}
module MnistLoader
  ( load_data_wrapper
  , TrainingData
  , ValidationData
  ) where

import           Codec.Compression.GZip (decompress)
import           Control.Applicative ((<$>))
import           Control.DeepSeq
import           Control.Monad (replicateM, unless)
import           Data.Binary.Get
import qualified Data.ByteString.Lazy as B
import           Data.Word (Word8)
import           Numeric.LinearAlgebra
import           System.FilePath.Posix ((</>))

type TrainingData   = [(Vector Double, Vector Double)]
type ValidationData = [(Vector Double, Word8)]


readImageFile :: FilePath -> IO [[Word8]]
readImageFile p = runGet parseImages <$> decompress <$> B.readFile p


readLabelFile :: FilePath -> IO [Word8]
readLabelFile p = runGet parseLabels <$> decompress <$> B.readFile p


parseImages :: Get [[Word8]]
parseImages = do
  magicNum <- getWord32be
  unless (magicNum == 2051) $ error $
    "Not an image file, wrong magic number " ++ show magicNum
  nImg <- fromIntegral <$> getWord32be
  nRow <- fromIntegral <$> getWord32be
  nCol <- fromIntegral <$> getWord32be
  img  <- replicateM nImg $ replicateM (nRow * nCol) getWord8
  unlessM isEmpty $ error "not all data consumed"
  return img


parseLabels :: Get [Word8]
parseLabels = do
  magicNum <- getWord32be
  unless (magicNum == 2049) $ error $
    "Not a label file, wrong magic number " ++ show magicNum
  n <- fromIntegral <$> getWord32be
  lbl <- replicateM n getWord8
  unlessM isEmpty $ error "not all data consumed"
  return lbl


load_data_wrapper :: FilePath -> IO (TrainingData, ValidationData)
load_data_wrapper p = do
  imgTraining    <- readImageFile $ p </> "train-images-idx3-ubyte.gz"
  imgValidation  <- readImageFile $ p </> "t10k-images-idx3-ubyte.gz"
  lblTraining    <- readLabelFile $ p </> "train-labels-idx1-ubyte.gz"
  lblValidation  <- readLabelFile $ p </> "t10k-labels-idx1-ubyte.gz"
  return ( (zip $!! normalizeImg imgTraining)   $!! toUnitVector lblTraining
         , (zip $!! normalizeImg imgValidation) $!! lblValidation
         )
  where
    normalizeImg = map $ fromList . map ((/ 256) . fromIntegral)
    toUnitVector = map (fromList . toUnitVector')
    toUnitVector' 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toUnitVector' 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    toUnitVector' 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    toUnitVector' 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    toUnitVector' 4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    toUnitVector' 5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    toUnitVector' 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    toUnitVector' 7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    toUnitVector' 8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    toUnitVector' 9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    toUnitVector' x = error $ "(toUnitVector " ++ show x ++ ") is undefined"


unlessM :: Monad m => m Bool -> m () -> m ()
unlessM mp f = (\p -> unless p f) =<< mp
