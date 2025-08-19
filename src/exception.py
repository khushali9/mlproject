import sys
import logging
from src.logger import logging 
#to able to save in log file
def error_message_detail(error,error_detail:sys):
    """
    This function returns a detailed error message.
    :param error: The exception object.
    :param error_detail: The sys module.
    :return: A string containing the error message.
    """

    _,_,exc_tb = error_detail.exc_info()#exc_tb is the traceback object
    # Extracting the file name and line number from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script: [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        """
        Custom exception class that inherits from the built-in Exception class.
        :param error_message: The error message to be displayed.
        :param error_detail: The sys module for detailed error information.
        """
        super().__init__(error_message) #cause inheriting from Exception
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        """
        Returns the string representation of the error message.
        :return: The error message.
        """
        return self.error_message #when we print the object, it will call this method and print the error message
    