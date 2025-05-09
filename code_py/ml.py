import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

attack_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def send_email_alert(attack_time):
    sender_email = "hvu74101@gmail.com"
    receiver_email = "haiv01154@gmail.com"
    password = "nujc itdg wqjq zdiq"

    subject = "Cảnh báo !!!"
    body = f"Phát hiện máy bạn bị tấn công đáng ngờ lúc: {attack_time} ,ấn link ... xem chi tiết. "

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email cảnh báo đã được gửi!")
    except Exception as e:
        print(f"Lỗi khi gửi email: {e}")

# Test gửi cảnh báo
attack_time1 = attack_time
send_email_alert(attack_time1)
