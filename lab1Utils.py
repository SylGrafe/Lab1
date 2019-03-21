#!/usr/bin/python3.5

import sys
import os
import sys
import datetime
import json
from json import JSONDecoder, JSONDecodeError
from json import encoder
import re
import numpy as np
import traceback
"""
from keras import losses
from keras import metrics
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from collections import namedtuple
encoder.FLOAT_REPR = lambda o: format(o, '.4f')



NOT_WHITESPACE = re.compile(r'[^\s]')
theDummyDumpName = "./benchMark.dummy_json"

"""
Contains functions and the Class  ConfigAndResults meant to be used
to dump on disk and read from disk history and params from a model
using json 

Remember 
dumps instance of ConfigAndResults  
but when reading info from disk will get  a dictionnary representation of a  instance ConfigAndResults and not a real  instance of   ConfigAndResults

Remember that this file must on the python path to be imported
"""

#################################################################################33
############################################# Class ConfigAndResults
class ConfigAndResults:
# The class represent the main information about a model and the training results 
# but as the info will be saved on disk there are almost no methods associated to the class
# instead   globals function    using  the dictionnary representation of the class wil lbe used

# the constructor
  def __init__(self,modelStruct, compInfo,histDict ,
          histParams , timeStamp, info="" ,h5 = ""  , testRes = None):
    self.modelStruct = modelStruct  # string to identify the model  layers
    self.compInfo = compInfo  # more info about compilation
    self.histDict = histDict # history.history
    self.histParams = histParams  # history.params
    self.timeStamp = timeStamp  
      # time at which the pgm which create a instance of this class was started
    self.info = info # if not empty more info to identify the test
    self.h5 = h5 # if not empty name of the saved model 
    self.testRes = testRes # if not empty test Resultat        
  
  def print_params (self):
    print (self.histParams)

# overwrite __str__ used when printing an instance of the class
  def __str__(self):
    return "(%s %s\n%s\n\n%s\n%s) " % (self.modelStruct , self.compInfo,self.histDict , 
    self.histParams , self.timeStamp)


  def toString (self):
    # print ("here in toString")
    # full trick to be able to round the values of the floats
    selfDict = round_floats(self.__dict__)
    selfNT = namedtuple("selfNT", selfDict.keys())(*selfDict.values())
    
    return "(%s %s\n%s\n\n%s\n%s) " % (selfNT.modelStruct , selfNT.compInfo,
    selfNT.histDict , selfNT.histParams , selfNT.timeStamp)



############################### global functions  #####################################
##################################################### function  round_floats(...)
def round_floats(o):
    if isinstance(o, float): return round(o, 4)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


################################################### function  dumpOnFile(...)
# dump an object on file using json
def dumpOnFile (someObj , theDumpFileName):
    # now let's  encode and  write  on disk
    print ("************ dumpOnFile () : append to file  %s  " % (theDumpFileName) )
    print ("\n Dumping object %s in the file  %s " % ( someObj.modelStruct , theDumpFileName))

    theDumpFile = open(theDumpFileName, 'a')
    json.dump(round_floats(someObj.__dict__),theDumpFile )

    # close the file, and your pickling is complete
    theDumpFile.close()

################################################### function plotHist(...)
# plot  acc (and val_acc if any ) in a subplot and 
# and loss  (and val_loss if any ) 
# given a instance of  ConfigAndResults or 
# a  namedTulpe representing an instance of  ConfigAndResults
def plotHist (someNamedTulpe ):

  try : 
    """
     self.modelStruct = x0  # string to identify the model  layers
    self.compInfo = x1  # more info about compilation
    self.histDict = x2 # history.history
    self.histParams = x3  # history.params
    self.timeStamp = x4  # time at which the pgm which create a instance of this class was started
    
    """
    theHistDict = someNamedTulpe.histDict
    theTitle  = someNamedTulpe.modelStruct + someNamedTulpe.info

    if not isinstance (theHistDict, dict) :
      print ("\n**** plotHist() failure expecting a dict and got: %s"  % (type(theHistDict)) )
      return 

    symbols = ['b' , 'bo']
    ii=0
    
    # try to plot acc
    ax = plt.subplot(2, 1, 1)
    ax.set_title(theTitle)
    for ki in theHistDict.keys():
      if ki.find('acc') != -1 :
        # print(ki)
        theValues = theHistDict[ki]
        #print ("\n\n type(theValues ) "  , type(theValues ) , "---> " , theValues)
        epochs = range(1, len(theValues) + 1)
        plt.plot(epochs, theValues, symbols[ii%2], label=ki)
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        ii+=1
    if (ii != 0 ):    
      plt.legend()
    
    ax = plt.subplot(2, 1, 2)
    jj= 0
    for ki in theHistDict.keys():
      if ki.find('loss') != -1 :
        # print(ki)
        theValues = theHistDict[ki]
        epochs = range(1, len(theValues) + 1)
        plt.plot(epochs, theValues, symbols[jj%2], label=ki)
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        jj+=1
    if (jj != 0 ):    
      plt.legend()
          
    # strange that you need to change hspace  and not wspace
    # to get  space enouth to see the tite of the second plot
    plt.subplots_adjust(hspace = 0.5)
    

    if (ii == 0 and jj== 0 ) :
      print ("plotHist() failure  there was neither acc or loss to plot ")
    else:
     plt.show()
      
  except Exception as e:
    print ("\n********** plotHist() fatal error: " , str(e))
  
################################################### function printAllFromFile(...)
def printAllFromFile (theDumpFileName):
  try : 

    if (not os.path.isfile(theDumpFileName)):
      print("printAllFromFile(), FATAL error %s is not a file"  %  ( theDumpFileName ) )
      return

    # now open a file for reading
    filePtr2 = open(theDumpFileName, 'r')
    document = filePtr2.read()
    # print (document)

    ii=0
    for obj in decode_stacked(document):
      ii+=1
      print(" %2d : %s\n" % (ii, obj) )
    print ("************************************************")

    # close the file, just for safety
    filePtr2.close()
  except Exception as e:
    print ("\n********** printAllFromFile() fatal error " % (theDumpFileName , str(e)))

################################################### function  printHeadersFromFile(...)
# read dump file and print part of each dump 
# remember that the results are all dictionnaries
def printHeadersFromFile (theDumpFileName):
  try : 

    if (not os.path.isfile(theDumpFileName)):
      print("printAllFromFile(), FATAL error %s is not a file "  %  ( theDumpFileName ) )
      return

    # now open a file for reading
    filePtr2 = open(theDumpFileName, 'r')
    document = filePtr2.read()
    """
    self.modelStruct = x0  # string to identify the model  layers
    self.compInfo = x1  # more info about compilation
    self.histDict = x2 # history.history
    self.histParams = x3  # history.params
    self.timeStamp = x4  # time at which the pgm which create a instance of this class was started
    self.info = info # if not empty more info to identify the test
    self.h5 = h5 # if not empty name of the saved model 
    """    

    ii=0
    for someDict in decode_stacked(document):
      # use named tulpe to access the dict as it would be real instance of ConfigAndResults 
      xxYY = namedtuple("XXYY", someDict.keys())(*someDict.values())
      ii+=1
      print(" %2d, ref:%s ,  model: %s , ,info: %s\n test[loss,acc]: %s\n" % 
      (ii, xxYY.timeStamp , xxYY.modelStruct , xxYY.info, xxYY.testRes ) )
    print ("************************************************")

    # close the file, just for safety
    filePtr2.close()
  except Exception as e:
    print ("\n********** printAllFromFile() %s fatal error %s" % (theDumpFileName , str(e)))


################################################### function getOneResFromFile(...)
# read in dumpfile the info about  the object referenced by refStr or ind
# return as a namedtuple  the first instance found
def getOneResFromFile (theDumpFileName , refStr= "" , ind=-1 ):
  try : 
    if (not os.path.isfile(theDumpFileName)):
      print("printAllFromFile(), FATAL error %s is not a file"  %  ( theDumpFileName ) )
      return

    # now open a file for reading
    filePtr2 = open(theDumpFileName, 'r')
    document = filePtr2.read()
    """
    self.modelStruct = x0  # string to identify the model  layers
    self.compInfo = x1  # more info about compilation
    self.histDict = x2 # history.history
    self.histParams = x3  # history.params
    self.timeStamp = x4  # time at which the pgm which create a instance of this class was started
    self.info = info # if not empty more info to identify the test
    self.h5 = h5 # if not empty name of the saved model 
    """    

    ii=0
    foundRes=False
    # remember that the info  read from the dump file are  dictionnaries
    for someDict in decode_stacked(document):
      # use named tulpe to access the dict as it would be real instance of ConfigAndResults 
      xxYY = namedtuple("XXYY", someDict.keys())(*someDict.values())
      if (ii == ind or xxYY.timeStamp == refStr ):
        foundRes=True
        print (" %s got record for  %d" % (refStr , ii)) 
        return xxYY
      ii+=1

    if not  foundRes:
      print ("getOneResFromFile() no results for file:%s ref:%s ind:%s" %
         ( theDumpFileName , refStr , ind ))
    # close the file, just for safety
    filePtr2.close()
    return None
  except Exception as e:
    print ("\n********** getOneResFromFile() %s fatal error %s" % (theDumpFileName , str(e)))
    return None


################################################### function printOneRes(...)
# expecting a namedtuple  similar to an instance of ConfigAndResults
# print some parts of the content

"""
  def __init__(self,modelStruct, compInfo,histDict ,
          histParams , timeStamp, info="" ,h5 = ""  , testRes = None):
    self.modelStruct = modelStruct  # string to identify the model  layers
    self.compInfo = compInfo  # more info about compilation
    self.histDict = histDict # history.history
    self.histParams = histParams  # history.params
    self.timeStamp = timeStamp  
      # time at which the pgm which create a instance of this class was started
    self.info = info # if not empty more info to identify the test
    self.h5 = h5 # if not empty name of the saved model 
    self.testRes = testRes # if not empty test Resultat        

"""
def printOneRes ( xxZZ ):
  try : 
        print (" %s, %s , %s " % (xxZZ.modelStruct ,xxZZ.compInfo , xxZZ.timeStamp)) 
        print (" testRes %s  " % (xxZZ.testRes  )) 
        print ("\nhistParams: %s " % (  xxZZ.histParams)) 
  except Exception as e:
    print ("\n********** printOneResFromFile()  fatal error %s" % ( str(e)))
    return None



################################################### function readAllFromFile ()
def readAllFromFile (theDumpFileName):
  try : 
    if (not os.path.isfile(theDumpFileName)):
      print("printAllFromFile(), FATAL error %s is not a file"  %  ( theDumpFileName ) )
      return

    theArr = []
    # now open a file for reading
    filePtr2 = open(theDumpFileName, 'r')
    document = filePtr2.read()
    # print (document)

    for obj in decode_stacked(document):
      theArr.append(obj)

    # close the file, just for safety
    filePtr2.close()
    return theArr

  except Exception as e:
    print ("\n********** readAllFromFile() fatal error " % (theDumpFileName , str(e)))




############################## function decode_stacked(...)
def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj


##################################### function doProceedUserInput

def doProceedUserInput (theDumpFileName):
  # get  user input
  # rawinput no data conversion
  myPrompt = "\nindStr [str] > " 
  print(myPrompt, end=' ')
  try:
    userStr= input()
    if (userStr == "" or userStr.lower().find('h') != -1 or 
             userStr.lower().find('?') != -1 ):
      theInfo ="""
      explore the content of the dumpfile %s
      if indStr== valid record indice       get info about the record
      if indStr== s     get list of all records 
      if indStr ==e     exit the Pgm
      if indStr==validindice and str==p   plot the history
      """ % ( theDumpFileName)
    
      print ( theInfo)
      return   
    if (userStr.lower().find('s') != -1 ):

      printHeadersFromFile (theDumpFileName)
      return    
    if (userStr.lower().find('e') == 0 ):
      print ("doProceedUserInput() exit")
      sys.exit(1)

    try:
      indStr , inputStr =  userStr.split()
    except ValueError:
      # maybe the user did not enter the string after the indice
      #print ("doProceedUserInput () here in ValueError ")
      #print ("doProceedUserInput invalid input")
      inputStr=" "
      indStr = userStr

    try:
      theInd=int ( indStr )
    except ValueError:
      # maybe the user did not enter the string after the indicise
      print ("doProceedUserInput invalid input %s should be an indice" % indStr)
      return

    print ("doProceedUserInput () userStr:%s " % (userStr))
    print ("doProceedUserInput () indStr:%s " % (indStr))


    someRes = getOneResFromFile (theDumpFileName , ind=theInd)
    if (someRes):
        # print ("type(someRes): " , type(someRes))
        printOneRes (someRes)
        if (inputStr.lower() == 'p'):
          plotHist(someRes)  
  except KeyboardInterrupt:
      print ("Bye , You hit control-C ")
      sys.exit(0)


  except ValueError:
    # maybe the user did not enter the string after the indicise
    print ("doProceedUserInput () here in ValueError ")
    print ("doProceedUserInput invalid input")

    inputStr=" "

  except KeyboardInterrupt:
    print ("Bye , You hit control-C ")
    sys.exit(0)
  except Exception as e:
      exceptVar = traceback.format_exc()
      print( "exception in doProceedUserInput()  " )
      print( exceptVar   )



############################################### main 
#  test create an object and  dump it in the benchmark

if __name__ == "__main__":

  doDump=False

  if doDump:
    """
  def __init__(self,modelStruct, compInfo,histDict ,
          histParams , timeStamp, info="" ,h5 = ""  , testRes = None):

    """
    dummyHist={'h1':"dummy" , 'h2':"dummy"}
    dummyParams={'p1':"dummy" , 'p2':"dummy"}
    dummyObj = ConfigAndResults ('dummyinfo' , "dummy compilering info", 
        dummyHist ,  dummyParams,
    datetime.datetime.now().strftime("%d%m at %H:%M:%S") )
    print  ("TEST TEST wil ltry to dump dummy object :%s" % (dummyObj)  )

    dumpOnFile (dummyObj , theDummyDumpName)

  # retrieve all data from disk
  print ("\n Try to retrieve content from  file %s  " %  theDumpFileName)

  printAllFromFile()

  print ("\n\n read allfrom file")
  theTests = readAllFromFile (theDummyDumpName)
  # print the first object
  theInd=0
  
   # Parse JSON into an object with attributes corresponding to dict keys.
  someElement = theTests[theInd]
  print ( "\ntheTests[%d] has type  %s " % (theInd , type(someElement)))

  if  isinstance (someElement , dict) :
    print ( "someElement['info']: " ,  someElement['info'])
    print ( "someElement['histDict']: " ,  someElement['histDict'])
    plotHist (someElement['histDict'] , someElement['info'])

  else :
    print ("failure %s is not a dict" % (type(someElement)))
   
print ("\n")


  


