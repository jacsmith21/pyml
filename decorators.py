import context
import parser


# noinspection PyPep8Naming
class param:
    param_instances = []
    config_fn_names = []

    def __init__(self, config_fn=None, **kwargs):
        """
        This is the Param decorator initializer. If there are no arguments given (ex `@Param`) then the config_fn is
        given as a positional parameter. If arguments are given (ex. `@Param(abstract=True)` then they are given this
        constructor in the `**kwargs` variable.

        :param config_fn: The configuration method.
        :param kwargs: The optional configuration arguments.
          `abstract`: A boolean value stating whether or not the config variable is abstract and should be implemented
          by a child class.

          `flag`: A boolean value stating whether or not the config variable is a flag. This is used to set the action
          of the parser.
        """
        self.abstract = kwargs.get('abstract', False)
        self.flag = kwargs.get('flag', False)

        if config_fn is not None:
            self.config_fn = config_fn
            self._init()

    def __call__(self, config_fn):
        """
        If function is called if arguments are given. It passes in the config function that wasn't given in the
        constructor.

        :param config_fn: The config method.
        :return: The param object.
        """
        self.config_fn = config_fn
        self._init()
        return self

    def _init(self):
        """
        This method is the initializer for the Param object. By default, we cache the values so that each config method
        is only run once. This allows for the creation of objects in config methods which keep an internal state (ex.
        the Label object) and should only be created once. Furthermore, this method sets overwritten to false and adds
        the argument to the parser.
        """
        param.param_instances.append(self)

        self.cache = None
        self.cached = False

        self.value = None
        self.overwritten = False

        self.config_fn_name = self.config_fn.__name__

        param.config_fn_names.append(self.config_fn_name)

        parser.add_argument('--{}'.format(self.config_fn_name), self.config_fn.__doc__, self.flag)

    def _has_test_param(self):
        """
        This method checkouts for a config method with the value `config_name_test` if the config function was called
        `config_name`.

        :return: Whether or not a config method also has an associated test method.
        """
        return '{}_test'.format(self.config_fn_name) in param.config_fn_names

    def __get__(self, instance, owner):
        """
        This method returns the appropriate value for the config variable. It performs multiple checks before calling
        the actual config method:
        1. Has the value been overwritten using the __set__ method?
        2. Are we in a test and is there a test value for the config variable?
        3. Did we provide a value from the command line?
        4. Is this config variable abstract?

        After completing those checks, we call the actual function and cache the value.

        :param instance: The instance of the Config object.
        :param owner: The owner.
        :return: The appropriate config value.
        """
        if self.overwritten:
            return self.value

        if context.in_test() and self._has_test_param():
            return instance.__getattribute__('{}_test'.format(self.config_fn_name))

        if parser.contains(self.config_fn_name):
            return parser.get(self.config_fn_name)

        if self.abstract:
            raise NotImplementedError(
                '{} is abstract and should be implemented by the child class.'.format(self.config_fn_name))

        if not self.cached:
            self.cache = self.config_fn(instance)
            self.cached = True

        return self.cache

    def __set__(self, instance, value):
        self.value = value
        self.overwritten = True

    @staticmethod
    def reset():
        """
        This is a static method that needs to be called each time a new Config instance is created as Param objects are
        not created every time a new Config object is created.

        :return:
        """
        for param_instance in param.param_instances:
            param_instance.cached = False
            param_instance.overwritten = False
