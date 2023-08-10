# coding: utf-8

import json
import hashlib
import logging
from requests import PreparedRequest

from redis import StrictRedis
from redis.connection import ConnectionPool
from redis.sentinel import Sentinel
from redis.sentinel import SentinelConnectionPool

from decouple import config


class RedisConnectorBase(object):
    conn = None
    print('###### RedisConnector ######')
    @classmethod
    def get_conn(cls):
        # Quando estiver rodando os testes nao usa o Redis como cache.
        teste = config('RUNNING_TESTS', cast=bool)
        redis_off = config('REDIS_OFF', cast=bool)
        if redis_off or teste:
            cls.conn = RedisDummy()

        if cls.conn is None:

            if config('REDIS_CACHE_ON', cast=bool):
                if config('REDIS_SENTINEL_BASED', cast=bool):
                    sentinel = Sentinel([
                        (host, config('DBAAS_SENTINEL_PORT', cast=int))
                        for host in config('DBAAS_SENTINEL_HOSTS').split(',')
                    ])
                    pool = SentinelConnectionPool(
                        config('DBAAS_SENTINEL_SERVICE_NAME'),
                        sentinel,
                        password=config('DBAAS_SENTINEL_PASSWORD'))
                else:
                    pool = ConnectionPool(host=config('REDIS_HOST'), port=config('REDIS_PORT', cast=int))
                cls.conn = StrictRedis(connection_pool=pool,
                                       max_connections=config('REDIS_MAX_CONN'),
                                       socket_timeout=config('REDIX_SOCKET_TIMEOUT'))
                logging.info('Recuperou conexao com RedisConnector (Redis ON)')
            else:
                cls.conn = RedisDummy()
                logging.info('Recuperou conexao com RedisDummy (Redis OFF)')

        return cls.conn


class RedisConnector(RedisConnectorBase):

    def __init__(self):
        self.conn = self.get_conn()

    def get(self, key):
        cache_obj = self.conn.get(key)
        if cache_obj:
            value = json.loads(cache_obj)
            return value, True
        return cache_obj, False

    def set(self, key, value, expiration=120):
        cache_obj = json.dumps(value)
        return self.conn.setex(key, expiration, cache_obj)

    def delete(self, key):
        return self.conn.delete(key)

    def flushall(self):
        return self.conn.flushall()

    def make_key(self, expiration, url, qs=None):
        pr = PreparedRequest()
        pr.prepare_url(url=url, params=qs)
        key = '%s:%s' % (expiration, hashlib.md5(pr.url.encode('utf-8')).hexdigest())  # #nohusky
        return key


class RedisDummy(object):
    """
    Dummy object to simulate StrictRedis behavior

    Used to mock access to Redis when caching is disabled.
    """
    print('entrou no RedisDummy')

    def get(self, *args, **kwargs):
        return None

    def setex(self, *args, **kwargs):
        return None

    def delete(self, *args, **kwargs):
        return None

    def flushall(self, *args, **kwargs):
        return None
