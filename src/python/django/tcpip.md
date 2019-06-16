# TCP/IP 分层

1. 应用层 

    决定向用户提供应用服务时的活动
    ( ftp, dns, http )

2. 传输层 

    对上层应用层提供处于网络连接中的两台计算机之间的数据传输
    TCP (transmission control protocol)/ UDP (user data protocol) 

3. 网络层
   
   处理网络上流动的数据包(数据包是网络传输最小数据单位), 规定了传输路径

4. 链路层
    
    处理连接网络的硬件部分

# IP ( internet protocol )

    作用 : 把数据包传输到对方
    ip地址 : 节点被分配到的地址
    mac地址 :  网卡固定地址

# TCP

    提供字节流服务( byte stream service)
    将大块数据分割成以报文段(segment)为单位的数据包进行管理
    确保数据送到, tcp协议采用三次握手
    发送端 发送带syn(synchronize)标字的数据包给对方
    接收端 返回带有syn/ack(acknowledgement)标志数据包 
    发送端 发送带ack的数据包

# DNS ( domain name system)

    提供域名到ip地址之间的解析服务

# URI (uniform resource identifier)

    uri 统一资源标识符
    url 统一资源定位符

# HTTP协议

## 持久连接

    只要任意一端没有明确提出断开连接,则保存连接

## 管线化(pipelining)

    同时并行发送多个请求

## 使用cookie 状态管理

    通过在请求和响应报文中写入cookie 来控制客户端状态

## HTTP 报文

    用于http协议交互的信息称为http报文
    报文分为报文首部和报文主体

## 状态码

    200 ok
    204 no content 响应报文不含实体主体
    206 partial content 

    301 moved permanently 永久重定向
    302 found 临时重定向
    303 see other
    304 not modified 服务器允许请求访问资源,但未满足条件
    307 temporary redirect

    400 bad request
    401 unauthorized
    403 forbidden
    404 not found

    500 internal server error
    503 service unavailables


## 数据转发程序 代理 网关 隧道

    HTTP通信时除了客户端和服务端,还有一些用于通信数据转发的应用程序

    代理 转发功能的应用程序
        缓存代理
        透明代理
    网关 转发其他服务器通信数据的服务器 
    隧道 保持客户端服务器通信连接的应用程序 使用ssl等加密手段进行通

## 缓存

    缓存服务器是代理服务器的一种

## http首部

    首部内容为客户端和服务端分别处理请求和响应提供所需要的信息  