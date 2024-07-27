import logging
from pythonjsonlogger import jsonlogger
import colorlog
from logging.handlers import (
    RotatingFileHandler,
    TimedRotatingFileHandler
)
import traceback
import sys
from typing import List

"""
LOGGINGS GUIDE: https://betterstack.com/community/guides/logging/how-to-start-logging-with-python/

The available log levels in the logging module are listed below in increasing order of severity:
    - DEBUG (10): used to log messages that are useful for debugging.
    - INFO (20): used to log events within the parameters of expected program behavior.
    - WARNING (30): used to log unexpected events which may impede future program function but not severe enough to be an error.
    - ERROR (40): used to log unexpected failures in the program. Often, an exception needs to be raised to avoid further failures, 
      but the program may still be able to run.
    - CRITICAL (50): used to log severe errors that can cause the application to stop running altogether.

The Formatter does what you'd expect; it helps with formatting the output of the logs.
The Handler specifies the log destination which could be the console, a file, an HTTP endpoint, and more. 
Filter objects provide sophisticated filtering capabilities for your Loggers and Handlers.

Adding contextual variables in the manner shown above can be done by using a formatting directive in the log message and passing 
variable data as arguments.
    - However, note that the logging module is optimized to use the % formatting style  and the use of f-strings might have an 
      extra cost.
name = "james"
browser = "firefox"
logger.info("user %s logged in with %s", name, browser)

fmt = logging.Formatter(
    "%(name)s: %(asctime)s | %(levelname)s | %(filename)s%(lineno)s | %(process)d | %(user)s | %(session_id)s >>> %(message)s"
)
ogger.info("Info message", extra={"user": "johndoe", "session_id": "abc123"})

Structured JSON logging in Python
most logs aren't read directly by humans anymore but processed with log aggregation tools first to quickly extract the insights 
required. With our current plain text format, most of these tools will be unable to automatically parse our logs (at least without 
the help of some complex regular expression). Therefore, we need to log in a standard structured format (such as JSON) to ensure 
that the logs are easily machine-parseable.
One of the advantages of using the python-json-logger library is that is allows you to add context to your logs through the extra 
property without needing to modify the log format.

Logging errors:
try:
    1 / 0
except ZeroDivisionError as e:
    logger.error(e, exc_info=True)
    logger.exception(e)
    logger.critical(e, exc_info=True)

Logging uncaught exceptions: https://betterstack.com/logs
Uncaught exceptions are caused by a failure to handle a thrown exception in the program with a try/catch block. When an such an 
exception occurs, the program terminates abruptly and an error message (traceback) is printed to the console.

It is helpful to log uncaught exceptions at the CRITICAL level so that you can identify the root cause of the problem and take 
appropriate action to fix it. If you're sending your logs to a log management tool like Logtail, you can configure alerting on such 
errors so that they are speedily resolved to prevent recurrence.

Automatically rotating log files: https://betterstack.com/community/guides/logging/how-to-manage-log-files-with-logrotate-on-ubuntu-20-04/
When you're logging to files, you need to be careful not to allow the file grow too large and consume a huge amount of disk space. 
By rotating log files, older logs can be compressed or deleted, freeing up space and reducing the risk of disk usage issues. 
Additionally, rotating logs helps to maintain an easily manageable set of log files, and can also be used to reduce the risk of 
sensitive information exposure by removing logs after a set period of time.

The RotatingFileHandler class takes the filename as before but also a few other properties. The most crucial of these are backupCount 
and maxBytes. The former determines how many backup files will be kept while the latter determines the maximum size of a log file 
before it is rotated. In this manner, each file is kept to a reasonable size and older logs don't clog up storage space unnecessarily.

The Python logging hierarchy
The Python logging hierarchy is a way of organizing loggers into a tree structure based on their names with the root logger at the top.
Each custom logger has a unique name, and loggers with similar names form a hierarchy. When a logger is created, it inherits log levels 
and handlers from the nearest ancestor that does if it doesn't have those settings on itself. This allows for fine-grained control over 
how log messages are handled.

The propagate argument in the Python logging module is used to control whether log messages should be passed up the logging hierarchy 
to parent loggers. By default, this argument is set to True, meaning that log messages are propagated up the hierarchy.
If the propagate argument is set to False for a particular logger, log messages emitted by that logger will not be passed up the 
hierarchy to its parent loggers. This allows for greater control over log message handling, as it allows you to prevent messages from 
being handled by parent loggers.

Best practices for logging in Python
    - Use meaningful logger names: Give loggers meaningful names that reflect their purpose, using dots as separators to create a 
      hierarchy. For example, a logger for a module could be named module.submodule. You can use the __name__ variable to achieve this 
      naming convention.
    - Avoid using the root logger: The root logger is a catch-all logger that can be difficult to manage. Instead, create specific 
      loggers for different parts of your application.
    - Set the appropriate log levels: For example, you can use WARNING in production and DEBUG in development or testing environments.
    - Centralize your logging configuration: Centralizing your logging configuration in a single location will make it much easier to 
      manage.
    - Aggregate your logs: Consider using a library like logtail-python  for centralizing your logs, as it provides advanced features 
      like centralized logging, aggregation, and alerting.
    - Include as much context as necessary: Ensure that relevant context surrounding the event being logged is included in the log 
      record. At a minimum, records should always include the severity level, the logger name, and the time the message was emitted.
    - Avoid logging sensitive information: Avoid logging sensitive information, such as passwords, security keys or user data, as it 
      could compromise the security of your application.
    - Test your logging configuration: Test your logging configuration in different scenarios to ensure that it behaves as expected 
      before deploying to production.
    - Rotate log files regularly: Regularly rotate log files to keep them from growing too large and becoming difficult to manage. 
      We recommend using Logrotate but you can also use the RotatingFileHandler or TimedRotatingFileHandler as demonstrated in this 
      article.
"""

class LevelFilter(logging.Filter):
    """
    LevelFilter can be used to implement a wide range of logging policies, such as:
        - Including only messages from certain parts of your application
        - Including only messages of a certain severity
        - Including only messages that contain certain text
    """
    replace_lvl = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, levels: List[str]) -> None:
        # Set levels
        self.levels = [
            lvl.replace(self.replace_lvl) for lvl in levels
        ]

    def filter(self, record) -> bool:
        # Filter levels
        if record.levelno in self.levels:
            return True

def get_logger(
    name: str = None, 
    level: str = None,
    txt_fmt: str = None,
    json_fmt: str = None,
    filter_lvls: List[str] = None,
    log_file: str = None,
    backup_count: int = None
) -> logging.Logger:
    # Define default name, level & formats
    if name is None:
        name = __name__
    if level is None:
        level = logging.INFO
    if json_fmt is None:
        json_fmt = "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(message)s"
    if txt_fmt is None:
        txt_fmt = "%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(lineno)s%(reset)s | %(log_color)s%(message)s%(reset)s"

    # Interpret level
    interpreted_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    if level in interpreted_levels.keys():
        level = interpreted_levels.get(level)

    # Instanciate the logger
    logging.getLogger().setLevel(level=level)
    logger = logging.getLogger(name) # .setLevel(level=level)
    logger.setLevel(level=level)

    # Build a ColoredFormatter (text)
    txt_formatter = colorlog.ColoredFormatter(fmt=txt_fmt)

    # Build Filters
    if filter_lvls is not None:
        level_filter = LevelFilter(levels=filter_lvls)
    else:
        level_filter = None

    # Add filters to logger
    if level_filter is not None:
        logger.addFilter(level_filter)

    # Add a StreamHandler to output log messages to the standard output (stdout)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level=level)
    stdout_handler.setFormatter(txt_formatter)

    if level_filter is not None:
        stdout_handler.addFilter(level_filter)

    logger.addHandler(stdout_handler)

    if log_file is not None:
        """
        FileHandler: vanilla file handler to output log messages to a file
        RotatingFileHandler:
            - The log_file will will be created and written to as before until it reaches 5 megabytes.
            - This process continues until we get to logs.txt.5. At that point, the oldest file (logs.txt.5) 
              gets deleted to make way for the newer logs.
        TimedRotatingFileHandler:
            - Rotate files once a week at Sundays, while keeping a maximum of backup_count backup files
        """
        # Assert that the log_file ends with .log
        assert log_file.endswith('.log')

        # Define log_dir
        log_dir = f'logs/{log_file}'

        # Add a FileHandler to output log messages to a file
        # file_handler = logging.FileHandler(log_file)
        # file_handler = RotatingFileHandler(log_file, backupCount=5, maxBytes=5000000)
        file_handler = TimedRotatingFileHandler(
            filename=log_dir, 
            backupCount=backup_count, 
            when="W6" # Sundays
        )
        file_handler.setLevel(level=level)

        # Build a JsonFormatter to customize the log message format (json)
        json_formatter = jsonlogger.JsonFormatter(
            fmt=json_fmt,
            rename_fields={
                "name": "logger_name",
                "asctime": "timestamp",
                "levelname": "severity", 
            }
        )
        file_handler.setFormatter(json_formatter)

        if level_filter is not None:
            file_handler.addFilter(level_filter)

        logger.addHandler(file_handler)

    # Add logging for uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Get the traceback as a string
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        logger.critical(tb_str, exc_info=(exc_type, exc_value, exc_traceback)) # "Uncaught exception"

    sys.excepthook = handle_exception

    return logger