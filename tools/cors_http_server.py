import argparse
import functools
import http.server
import socketserver


class CorsRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8123)
    parser.add_argument('--directory', default='.')
    args = parser.parse_args()

    handler = functools.partial(CorsRequestHandler, directory=args.directory)
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as server:
        print(f'Serving {args.directory} on http://{args.host}:{args.port} with CORS enabled')
        server.serve_forever()


if __name__ == '__main__':
    main()