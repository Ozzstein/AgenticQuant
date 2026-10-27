import json

import pytest

from src.operations.notifications import (
    EmailChannel,
    NotificationPlan,
    TelegramChannel,
    dispatch_notifications,
)
from src.operations.store import OperationError, OperationsStore


def alert(tmp_path):
    store=OperationsStore(tmp_path/'ops.sqlite3')
    identity=store.alert(run_id='real-fixture-run',severity='critical',code='reconciliation_failed',payload={'cash_difference':'1.23'})
    return store,identity


def test_missing_credentials_are_not_delivery_and_telegram_receipt_is_durable(tmp_path,monkeypatch):
    store,identity=alert(tmp_path)
    channel=TelegramChannel(chat_id='123')
    plan=NotificationPlan(channels=(channel,))
    monkeypatch.delenv(channel.token_env,raising=False)
    assert dispatch_notifications(store,plan)[0]['status']=='unavailable'
    requests=[]
    class Connection:
        def __init__(self,host,**options):
            assert host=='api.telegram.org' and options['timeout']==15
        def request(self,method,path,*,body,headers):
            requests.append((method,path,json.loads(body),headers))
        def getresponse(self):
            class Response:
                status=200
                def read(self,limit):
                    assert limit==65537
                    return b'{"ok":true,"result":{"message_id":42,"chat":{"id":123}}}'
            return Response()
        def close(self):
            pass
    monkeypatch.setenv(channel.token_env,'123:fixture-secret')
    monkeypatch.setattr('src.operations.notifications.http.client.HTTPSConnection',Connection)
    result=dispatch_notifications(store,plan)
    assert result[0]['status']=='accepted' and result[0]['receipt']['message_id']==42
    assert requests[0][2]['chat_id']=='123' and 'cash_difference' in requests[0][2]['text']
    assert 'parse_mode' not in requests[0][2]
    assert dispatch_notifications(OperationsStore(store.path),plan)==[]
    assert len(requests)==1 and 'fixture-secret' not in json.dumps(result)
    assert not store.pending_alerts('telegram:'+channel.content_hash())


def test_ambiguous_http_never_retries_until_explicit_review(tmp_path,monkeypatch):
    store,identity=alert(tmp_path)
    channel=TelegramChannel(chat_id='123')
    key='telegram:'+channel.content_hash()
    monkeypatch.setenv(channel.token_env,'123:fixture-secret')
    calls=[]
    class TimeoutConnection:
        def __init__(self,*args,**kwargs):
            pass
        def request(self,*args,**kwargs):
            calls.append(1)
        def getresponse(self):
            raise TimeoutError('secret response must never escape')
        def close(self):
            pass
    monkeypatch.setattr('src.operations.notifications.http.client.HTTPSConnection',TimeoutConnection)
    plan=NotificationPlan(channels=(channel,))
    assert dispatch_notifications(store,plan)[0]['status']=='uncertain'
    assert dispatch_notifications(store,plan)==[] and calls==[1]
    with pytest.raises(OperationError):
        store.resolve_delivery(identity,key,expected_attempt=2,outcome='retry_authorized',evidence={'review':'fixture'})
    receipt=store.resolve_delivery(identity,key,expected_attempt=1,outcome='retry_authorized',evidence={'operator':'fixture','reason':'verified not accepted'})
    assert store.artifact(receipt)['basis']=='operator_review'
    assert dispatch_notifications(store,plan)[0]['status']=='uncertain' and calls==[1,1]
    store.resolve_delivery(identity,key,expected_attempt=2,outcome='abandoned',evidence={'reason':'manual escalation'})
    assert dispatch_notifications(store,plan)==[]


def test_smtp_requires_tls_and_confirmed_recipient_acceptance(tmp_path,monkeypatch):
    store,_=alert(tmp_path)
    channel=EmailChannel(host='smtp.example.test',sender='sender@example.test',recipients=('recipient@example.test',))
    monkeypatch.setenv(channel.username_env,'fixture-user')
    monkeypatch.setenv(channel.password_env,'fixture-password')
    actions=[]
    class SMTP:
        def __init__(self,host,port,timeout):
            assert (host,port,timeout)==('smtp.example.test',587,15)
        def ehlo(self):
            actions.append('ehlo')
        def starttls(self,context):
            assert context.check_hostname
            actions.append('tls')
        def login(self,user,password):
            assert actions==['ehlo','tls','ehlo']
            assert (user,password)==('fixture-user','fixture-password')
            actions.append('login')
        def send_message(self,message,from_addr,to_addrs):
            assert message['Message-ID'].endswith('@quantagentlab.local>')
            assert from_addr==channel.sender and to_addrs==list(channel.recipients)
            actions.append('send')
            return {}
        def quit(self):
            actions.append('quit')
    monkeypatch.setattr('src.operations.notifications.smtplib.SMTP',SMTP)
    results=dispatch_notifications(store,NotificationPlan(channels=(channel,)))
    assert results[0]['status']=='accepted'
    assert actions==['ehlo','tls','ehlo','login','send','quit']
    assert 'fixture-password' not in json.dumps(results)


def test_email_header_injection_and_secret_in_configuration_rejected():
    with pytest.raises(ValueError):
        EmailChannel(host='smtp.example.test',sender='sender@example.test\nBcc: attacker@example.test',recipients=('recipient@example.test',))
    with pytest.raises(ValueError):
        TelegramChannel(chat_id='123',token_env='123:literal-token')
