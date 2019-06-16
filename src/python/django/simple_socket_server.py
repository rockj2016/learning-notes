import socket, time, requests, errno, select
import multiprocessing,threading

EOL1 = b'\n\n'
EOL2 = b'\n\r\n'
body = '''hello world'''
response_params = [
    'HTTP/1.0 200 OK',
    'DATA: 2019 0615',
    '......',
    body
]

response = '\r\n'.join(response_params)


def handle_connection(conn,addr):
    time.sleep(1)
    request = b''

    ready_to_read, ready_to_write, in_error = \
        select.select(
            conn,
            [],
            [],
            )
    print(ready_to_read)
    while EOL1 not in request and EOL2 not in request:
        request += conn.recv(1024)
    # print(request)
    conn.send(response.encode())
    conn.close()


def main():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(('127.0.0.1', 8000))
    serversocket.listen(5)
    serversocket.setblocking(0)
    print('127.0.0.1:8000')

    try:
        i = 0
        while True:
            i += 1
            try:
                conn,address = serversocket.accept()
            except socket.error as e:
                # if e.args[0] != errno.EAGAIN:
                #     raise
                continue
            t = threading.Thread(target=handle_connection,
                                 args=(conn, address),
                                 name=f'thread {i}')
            t.start()
            # print(conn)
            # print(address)
            handle_connection(conn,address)
    finally:
        serversocket.close()


def test_get():
    for i in range(5):
        res = requests.get('http://127.0.0.1:8000/')
        print(f'res status_code {res.status_code}')


if __name__ == '__main__':
    p = multiprocessing.Process(target=main, args=())
    p.start()
    # main()
    time.sleep(2)
    print('start_request')
    test_get()