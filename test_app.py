import io
import app as project

def test_home_route():
    client=project.app.test_client()
    assert client.get('/').status_code==200

def test_upload_and_preview():
    client=project.app.test_client()
    data={'dataset':(io.BytesIO(b'name,value\nA,10\nB,20\n'),'tiny.csv')}
    response=client.post('/upload',data=data,content_type='multipart/form-data',follow_redirects=True)
    assert response.status_code==200
    assert b'Data preview' in response.data
