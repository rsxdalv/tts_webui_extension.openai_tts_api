import argparse
import logging

logger = logging.getLogger(__name__)

# Registration config populated by _parse_args() before the server starts.
_reg = None


def _on_startup():
    if not _reg:
        return
    try:
        from .register import register
        register(**_reg)
        logger.info('[server] Registered as %r', _reg['model'])
    except Exception as exc:
        logger.warning('[server] Auto-registration failed: %s', exc)


def _on_shutdown():
    if not _reg:
        return
    try:
        from .register import unregister
        unregister(_reg['model'],
                   host=_reg.get('host', 'http://localhost:7778'),
                   main_api_key=_reg.get('main_api_key'))
        logger.info('[server] Unregistered %r', _reg['model'])
    except Exception as exc:
        logger.warning('[server] Auto-unregistration failed: %s', exc)


def _build_app(api_key=None):
    from .harness import create_app
    from .tts import get_voices, tts
    return create_app(tts, get_voices, api_key=api_key,
                      on_startup=_on_startup, on_shutdown=_on_shutdown)


def _parse_args():
    global _reg
    p = argparse.ArgumentParser(description='Proxied TTS server')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=12345)
    p.add_argument('--api-key', default=None, dest='api_key')
    p.add_argument('--model', default=None,
                   help='Register as this model name on startup')
    p.add_argument('--main-server', default='http://localhost:7778',
                   dest='main_server')
    p.add_argument('--main-api-key', default=None, dest='main_api_key')
    args, _ = p.parse_known_args()
    if args.model:
        _reg = dict(model=args.model,
                    url=f'http://localhost:{args.port}',
                    api_key=args.api_key,
                    host=args.main_server,
                    main_api_key=args.main_api_key)
    return args


if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    app = _build_app(api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port)
