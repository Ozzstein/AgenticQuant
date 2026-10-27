"""Bounded notification delivery through the durable operating outbox.

Transport acceptance is not recipient delivery. Credentials are read only from
named environment variables and never appear in configuration, receipts or errors.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime, parseaddr
from typing import Literal

from pydantic import Field, field_validator

from src.domain.common import NonEmptyString, VersionedContract
from src.operations.store import OperationError, OperationsStore, canonical, digest


def _environment(name):
    if not re.fullmatch(r'[A-Z][A-Z0-9_]{0,127}',name):
        raise ValueError('credential reference must name an environment variable')
    return name


class TelegramChannel(VersionedContract):
    kind: Literal['telegram'] = 'telegram'
    chat_id: NonEmptyString
    token_env: NonEmptyString = 'QUANTAGENT_TELEGRAM_TOKEN'
    timeout_seconds: float = Field(default=15,gt=0,le=60)

    _credential = field_validator('token_env')(_environment)


class EmailChannel(VersionedContract):
    kind: Literal['email'] = 'email'
    host: NonEmptyString
    port: int = Field(default=587,ge=1,le=65535)
    security: Literal['starttls','tls'] = 'starttls'
    sender: NonEmptyString
    recipients: tuple[NonEmptyString,...] = Field(min_length=1,max_length=20)
    username_env: NonEmptyString = 'QUANTAGENT_SMTP_USER'
    password_env: NonEmptyString = 'QUANTAGENT_SMTP_PASSWORD'
    timeout_seconds: float = Field(default=15,gt=0,le=60)

    _credentials = field_validator('username_env','password_env')(_environment)

    @field_validator('sender','recipients')
    @classmethod
    def addresses(cls,value):
        for address in (value,) if isinstance(value,str) else value:
            if '\r' in address or '\n' in address or parseaddr(address)[1] != address or '@' not in address:
                raise ValueError('explicit plain email address required')
        return value


class NotificationPlan(VersionedContract):
    channels: tuple[TelegramChannel | EmailChannel,...] = Field(min_length=1,max_length=10)
    max_per_dispatch: int = Field(default=20,ge=1,le=100)


def alert_text(alert: dict) -> str:
    # Plain text only: source text is never interpreted as Telegram HTML/Markdown.
    title = f"QuantAgentLab {alert['severity'].upper()}: {alert['code']}"
    body = json.dumps(alert['payload'],sort_keys=True,ensure_ascii=False,allow_nan=False)
    return f"{title}\nRun: {alert.get('run_id') or 'system'}\n{body}\nAlert: {digest(canonical(alert))}"


class NotificationTransport:
    def __init__(self, channel: TelegramChannel | EmailChannel):
        self.channel = channel

    def ready(self) -> bool:
        names = (self.channel.token_env,) if isinstance(self.channel,TelegramChannel) else (
            self.channel.username_env,self.channel.password_env)
        return all(bool(os.environ.get(name)) for name in names)

    def __call__(self, alert: dict) -> dict:
        if not self.ready():
            raise OperationError('notification credential is unavailable')
        try:
            receipt = self._telegram(alert) if isinstance(self.channel,TelegramChannel) else self._email(alert)
        except Exception:
            # HTTP/SMTP exceptions can contain URLs, credentials or server text.
            # No automatic retry: the outbox retains an uncertain delivery.
            raise OperationError('notification transport did not establish acceptance') from None
        return {**receipt,'channel_hash':self.channel.content_hash(),
                'observed_at':datetime.now(timezone.utc).isoformat(),'semantics':'transport_acceptance'}

    def _telegram(self,alert):
        settings = self.channel
        token = os.environ[settings.token_env]
        if not re.fullmatch(r'[0-9]+:[A-Za-z0-9_-]+',token):
            raise OperationError('invalid Telegram credential format')
        text = alert_text(alert)
        if len(text) > 4000:
            text = text[:3800]+'\n[Full evidence retained in operating store]\nAlert: '+digest(canonical(alert))
        connection = http.client.HTTPSConnection('api.telegram.org',timeout=settings.timeout_seconds,
                                                 context=ssl.create_default_context())
        try:
            connection.request('POST','/bot'+token+'/sendMessage',body=canonical({'chat_id':settings.chat_id,'text':text}),
                               headers={'Content-Type':'application/json'})
            response = connection.getresponse()
            payload = response.read(65537)
            if response.status != 200 or len(payload) > 65536:
                raise OperationError('Telegram response was not bounded successful acceptance')
            value = json.loads(payload)
            result = value.get('result',{})
            if value.get('ok') is not True or not isinstance(result.get('message_id'),int):
                raise OperationError('Telegram acceptance receipt missing')
            chat = result.get('chat',{})
            if str(chat.get('id')) != settings.chat_id and '@'+str(chat.get('username','')) != settings.chat_id:
                raise OperationError('Telegram accepted a different destination')
            return {'accepted':True,'transport':'telegram','message_id':result['message_id']}
        finally:
            connection.close()

    def _email(self,alert):
        settings = self.channel
        identity = digest(canonical(alert))
        message = EmailMessage()
        message['From'],message['To'] = settings.sender,', '.join(settings.recipients)
        message['Date'] = format_datetime(datetime.now(timezone.utc))
        message['Message-ID'] = f'<{identity}@quantagentlab.local>'
        message['Subject'] = f"QuantAgentLab {alert['severity']}: {alert['code']}"
        message.set_content(alert_text(alert))
        tls = ssl.create_default_context()
        client = (smtplib.SMTP_SSL(settings.host,settings.port,timeout=settings.timeout_seconds,context=tls)
                  if settings.security == 'tls' else smtplib.SMTP(settings.host,settings.port,timeout=settings.timeout_seconds))
        try:
            client.ehlo()
            if settings.security == 'starttls':
                client.starttls(context=tls)
                client.ehlo()
            client.login(os.environ[settings.username_env],os.environ[settings.password_env])
            refused = client.send_message(message,from_addr=settings.sender,to_addrs=list(settings.recipients))
            if refused:
                raise OperationError('one or more SMTP recipients were refused')
            return {'accepted':True,'transport':'smtp','message_id':message['Message-ID'],
                    'recipient_count':len(settings.recipients)}
        finally:
            # A failure after accepted DATA must not erase the measured receipt.
            try:
                client.quit()
            except (OSError,smtplib.SMTPException):
                client.close()


def dispatch_notifications(store: OperationsStore, plan: NotificationPlan) -> list[dict]:
    """Send only explicitly configured channels; missing access does not claim sent."""
    results = []
    remaining = plan.max_per_dispatch
    for channel in plan.channels:
        identity = channel.kind+':'+channel.content_hash()
        transport = NotificationTransport(channel)
        if not transport.ready():
            results.append({'channel':identity,'status':'unavailable'})
            continue
        for alert in store.pending_alerts(identity)[:remaining]:
            try:
                receipt = store.deliver(alert['id'],identity,transport)
                results.append({'channel':identity,'alert_id':alert['id'],'status':'accepted','receipt':receipt})
            except OperationError:
                results.append({'channel':identity,'alert_id':alert['id'],'status':'uncertain'})
            remaining -= 1
        if remaining == 0:
            break
    return results
