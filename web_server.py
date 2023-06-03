"""
Web-server for an automatic translation service.
Usage:
fetch('http://localhost:8080', {
    method: "POST", 
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({txt: `How are you?`})
}).then( resp => resp.json() ).then( data => console.log(data) );
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import json

from transformers import pipeline

hostName = "localhost"
serverPort = 8080

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-hu")

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = parse_qs(urlparse(self.path).query)
        print(parsed_path)

        if 'txt' not in parsed_path or parsed_path['txt'] == None:
            response = pipe("No text found!")
        else:
            response = pipe(parsed_path['txt'])

        self.do_Response(response[0]['translation_text'])

    def do_POST(self):
        content_type = str(self.headers.get('Content-Type'))
        if content_type != 'application/json':
            self.do_Response(
                json.dumps({'error': 'please send a json!'}))
            return

        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)

        body = json.loads(post_body)

        if 'txt' not in body or body['txt'] == None:
            self.do_Response(
                json.dumps({'error': 'no txt key found in the json!'}))
            return
        else:
            response = [{}]
            if content_len > 256:
                content = body['txt'].split('. ')
                parts = []
                for part in content:
                    parts.append(pipe(part)[0]['translation_text'])
                response[0]['translation_text'] = '. '.join(parts)
            else:
                response = pipe(body['txt'])

        self.do_Response(
            json.dumps(response[0]['translation_text'])
        )

    def do_Response(self, response, type = "text/html"):
        self.send_response(200)
        self.send_header("Content-type", type)
        self.end_headers()
        
        self.wfile.write(bytes(response, "utf-8"))


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
