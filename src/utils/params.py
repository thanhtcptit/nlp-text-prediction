import os
import copy
import json
import boto3
import shutil
import inspect
import logging
import requests
import tempfile

from tqdm import tqdm
from pathlib import Path
from hashlib import sha256
from functools import wraps
from overrides import overrides
from urllib.parse import urlparse
from collections import defaultdict
from botocore.exceptions import ClientError
from collections import MutableMapping, OrderedDict
from typing import Any, Dict, List, Set, TypeVar, Type, Union, cast, Optional, Tuple, IO, Callable


try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:
    def evaluate_file(filename: str, **_kwargs) -> str:
        logger.warning(f"_jsonnet not loaded, treating {filename} as json")
        with open(filename, 'r') as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs) -> str:
        logger.warning(f"_jsonnet not loaded, treating snippet as json")
        return expr

logger = logging.getLogger(__name__)

T = TypeVar('T')

_NO_DEFAULT = inspect.Parameter.empty

CACHE_ROOT = Path(os.getenv('CACHE_ROOT', Path.home() / '.cache_root'))
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")


class ConfigurationError(Exception):
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def url_to_filename(url: str, etag: str = None) -> str:
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    url_or_filename = os.path.expanduser(url_or_filename)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif parsed.scheme == '':
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func: Callable):
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url: str) -> Optional[str]:
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url: str, temp_file: IO) -> None:
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url: str, cache_dir: str = None) -> str:
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    os.makedirs(cache_dir, exist_ok=True)

    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s",
                        url, temp_file.name)

            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            temp_file.flush()
            temp_file.seek(0)

            logger.info("copying %s to cache at %s",
                        temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path



def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise ConfigurationError("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise ConfigurationError("flattened dictionary is invalid")
        else:
            curr_dict[parts[-1]] = value

    return unflat


def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            merged[key] = with_fallback(preferred_value, fallback_value)
        else:
            merged[key] = copy.deepcopy(preferred_value)

    return merged


def parse_overrides(serialized_overrides: str) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars = dict(os.environ)
        return unflatten(json.loads(evaluate_snippet("", serialized_overrides, ext_vars=ext_vars)))
    else:
        return {}


class Params(MutableMapping):
    DEFAULT = object()

    def __init__(self,
                 params: Dict[str, Any],
                 history: str = "",
                 loading_from_archive: bool = False,
                 files_to_archive: Dict[str, str] = None) -> None:
        self.params = _replace_none(params)
        self.history = history
        self.loading_from_archive = loading_from_archive
        self.files_to_archive = {} if files_to_archive is None else files_to_archive

    def add_file_to_archive(self, name: str) -> None:
        if not self.loading_from_archive:
            self.files_to_archive[f"{self.history}{name}"] = cached_path(
                self.get(name))

    @overrides
    def pop(self, key: str, default: Any = DEFAULT) -> Any:
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise ConfigurationError(
                    "key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.pop(key, default)
        if not isinstance(value, dict):
            logger.info(self.history + key + " = " + str(value))
        return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any = DEFAULT) -> int:
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any = DEFAULT) -> float:
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any = DEFAULT) -> bool:
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == "true":
            return True
        elif value == "false":
            return False
        else:
            raise ValueError("Cannot convert variable to bool: " + value)

    @overrides
    def get(self, key: str, default: Any = DEFAULT):
        if default is self.DEFAULT:
            try:
                value = self.params.get(key)
            except KeyError:
                raise ConfigurationError(
                    "key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any],
                   default_to_first_choice: bool = False) -> Any:
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        if value not in choices:
            key_str = self.history + key
            message = '%s not in acceptable choices for %s: %s' % (
                value, key_str, str(choices))
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet=False):
        if quiet:
            return self.params

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(history + key + " = " + str(value))

        logger.info("Converting Params object to dict; logging of default "
                    "values will not occur when dictionary parameters are "
                    "used subsequently.")
        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(self.params, self.history)
        return self.params

    def as_flat_dict(self):
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value

        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> 'Params':
        return Params(copy.deepcopy(self.params))

    def assert_empty(self, class_name: str):
        if self.params:
            raise ConfigurationError(
                "Extra parameters passed to {}: {}".format(class_name, self.params))

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value,
                          history=new_history,
                          loading_from_archive=self.loading_from_archive,
                          files_to_archive=self.files_to_archive)
        if isinstance(value, list):
            value = [self._check_is_dict(
                new_history + '.list', v) for v in value]
        return value

    @staticmethod
    def from_file(params_file: str, params_overrides: str = "", ext_vars: dict = None) -> 'Params':
        if ext_vars is None:
            ext_vars = {}

        params_file = cached_path(params_file)
        ext_vars = {**dict(os.environ), **ext_vars}

        file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))

        overrides_dict = parse_overrides(params_overrides)
        param_dict = with_fallback(
            preferred=overrides_dict, fallback=file_dict)

        return Params(param_dict)

    def to_file(self, params_file: str, preference_orders: List[List[str]] = None) -> None:
        with open(params_file, "w") as handle:
            json.dump(self.as_ordered_dict(
                preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: List[List[str]] = None) -> OrderedDict:
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(["dataset_reader", "iterator", "model",
                                      "train_data_path", "validation_data_path", "test_data_path",
                                      "trainer", "vocabulary"])
            preference_orders.append(["type"])

        def order_func(key):
            order_tuple = [order.index(key) if key in order else len(
                order) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            result = OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(
                    val, order_func) if isinstance(val, dict) else val
            return result

        return order_dict(params_dict, order_func)


def pop_choice(params: Dict[str, Any],
               key: str,
               choices: List[Any],
               default_to_first_choice: bool = False,
               history: str = "?.") -> Any:
    value = Params(params, history).pop_choice(
        key, choices, default_to_first_choice)
    return value


def _replace_none(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key in dictionary.keys():
        if dictionary[key] == "None":
            dictionary[key] = None
        elif isinstance(dictionary[key], dict):
            dictionary[key] = _replace_none(dictionary[key])
    return dictionary


def takes_arg(obj, arg: str) -> bool:
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return arg in signature.parameters


def remove_optional(annotation: type):
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and isinstance(None, args[1]):
        return args[0]
    else:
        return annotation


def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}

    for name, param in signature.parameters.items():
        if name == "self":
            continue

        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        default = param.default
        optional = default != _NO_DEFAULT

        if name in extras:
            kwargs[name] = extras[name]

        elif hasattr(annotation, 'from_params'):
            if name in params:
                subparams = params.pop(name)

                if takes_arg(annotation.from_params, 'extras'):
                    subextras = extras
                else:
                    subextras = {k: v for k, v in extras.items(
                    ) if takes_arg(annotation.from_params, k)}

                if isinstance(subparams, str):
                    kwargs[name] = annotation.by_name(subparams)()
                else:
                    kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif not optional:
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")
            else:
                kwargs[name] = default

        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if optional
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if optional
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if optional
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if optional
                            else params.pop_float(name))

        elif origin in (Dict, dict) and len(args) == 2 and hasattr(args[-1], 'from_params'):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)

            kwargs[name] = value_dict

        elif origin in (List, list) and len(args) == 1 and hasattr(args[0], 'from_params'):
            value_cls = annotation.__args__[0]

            value_list = []

            for value_params in params.pop(name, Params({})):
                value_list.append(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = value_list

        elif origin in (Tuple, tuple) and all(hasattr(arg, 'from_params') for arg in args):
            value_list = []

            for value_cls, value_params in zip(annotation.__args__, params.pop(name, Params({}))):
                value_list.append(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = tuple(value_list)

        elif origin in (Set, set) and len(args) == 1 and hasattr(args[0], 'from_params'):
            value_cls = annotation.__args__[0]

            value_set = set()

            for value_params in params.pop(name, Params({})):
                value_set.add(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = value_set

        else:
            if optional:
                kwargs[name] = params.pop(name, default)
            else:
                kwargs[name] = params.pop(name)

    params.assert_empty(cls.__name__)
    return kwargs


class Registrable:
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} "
                    f"and extras {extras}")

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({"type": params})

        registered_subclasses = Registrable._registry.get(cls)

        if registered_subclasses is not None:
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice("type",
                                       choices=as_registrable.list_available(),
                                       default_to_first_choice=default_to_first_choice)
            subclass = registered_subclasses[choice]
            if not takes_arg(subclass.from_params, 'extras'):
                extras = {k: v for k, v in extras.items() if takes_arg(subclass.from_params, k)}

            return subclass.from_params(params=params, **extras)
        else:
            if cls.__init__ == object.__init__:
                kwargs: Dict[str, Any] = {}
            else:
                kwargs = create_kwargs(cls, params, **extras)
            return cls(**kwargs)

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                    name, cls.__name__, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise ConfigurationError("%s is not a registered name for %s" % (name, cls.__name__))
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ConfigurationError(message)
        else:
            return [default] + [k for k in keys if k != default]
